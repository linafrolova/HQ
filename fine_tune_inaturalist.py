import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import py7zr
from pathlib import Path


def extract_archives(train_7z, test_7z, output_dir):
    """Extract train and test archives if not already extracted."""
    os.makedirs(output_dir, exist_ok=True)
    for archive in [train_7z, test_7z]:
        if not Path(archive).exists():
            raise FileNotFoundError(f"{archive} not found")
        with py7zr.SevenZipFile(archive, mode='r') as z:
            z.extractall(path=output_dir)


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df['name'] = df['name'].astype(str) + '.jpg'
    return dict(zip(df['name'], df['invasive']))


def prepare_dataset(image_dir, labels, batch_size=32, img_size=(299, 299)):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    y = [labels.get(Path(p).name, 0) for p in image_paths]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, y))

    def _process(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes=2):
    url = "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"
    feature_extractor = hub.KerasLayer(url, input_shape=(299, 299, 3), trainable=False)
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    train_archive = 'train.7z'
    test_archive = 'test.7z'
    labels_csv = 'train_labels.csv'
    data_dir = 'data'

    if not Path(data_dir).exists():
        extract_archives(train_archive, test_archive, data_dir)

    train_dir = os.path.join(data_dir, 'train')
    labels = load_labels(labels_csv)

    dataset = prepare_dataset(train_dir, labels)
    model = build_model()
    model.fit(dataset, epochs=5)
    model.save('invasive_model.h5')


if __name__ == '__main__':
    main()
