"""Entry point for the AI letter recognition application."""

from interface import DrawingInterface
from model import load_model, predict_letter


def main() -> None:
    model = load_model()

    def callback(img):
        return predict_letter(model, img)

    app = DrawingInterface(callback)
    app.run()


if __name__ == "__main__":
    main()
