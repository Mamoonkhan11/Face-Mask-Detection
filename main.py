# Main file to run the face mask detection training pipeline

from src.data_preprocessing import load_data
from src.model import build_model
from src.utils import plot_training, save_model_summary


def main():
    print("\n--> Loading dataset... \n ")
    train_data, test_data = load_data(img_size=(128,128), batch_size=32)

    print("--> Building MobileNetV2 model... \n")
    model = build_model(input_shape=(128,128,3))

    print("--> Training model...\n")
    hist = model.fit(
        train_data,
        validation_data=test_data,
        epochs=10
    )

    print("--> Saving model...\n")
    model.save("outputs/mask_detector_model.keras")

    save_model_summary(model)
    plot_training(hist)

    print(" Training finished successfully!")


if __name__ == "__main__":
    main()