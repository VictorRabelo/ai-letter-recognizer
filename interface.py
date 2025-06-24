import io
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

class DrawingInterface:
    """Tkinter interface for drawing and recognizing letters."""

    def __init__(self, predict_callback):
        self.predict_callback = predict_callback
        self.window = tk.Tk()
        self.window.title("AI Letter Recognizer")

        self.canvas_width = 200
        self.canvas_height = 200
        self.stroke_width = 8

        self.canvas = tk.Canvas(
            self.window, width=self.canvas_width, height=self.canvas_height, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        self.predict_label = tk.Label(self.button_frame, text="Draw a letter")
        self.predict_label.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(
            self.button_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_button.pack(side=tk.LEFT)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_draw_end)
        self.last_x = None
        self.last_y = None

        # Image for internal representation
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=self.stroke_width,
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.draw_image.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="black",
                width=self.stroke_width,
            )
        self.last_x = event.x
        self.last_y = event.y

    def on_draw_end(self, event):
        self.last_x = None
        self.last_y = None
        img = self.capture_image()
        if self.predict_callback:
            letter, confidence = self.predict_callback(img)
            if letter:
                self.predict_label.config(
                    text=f"Prediction: {letter} ({confidence:.2f})"
                )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle(
            [0, 0, self.canvas_width, self.canvas_height], fill="white"
        )
        self.predict_label.config(text="Draw a letter")

    def capture_image(self):
        # Crop to content
        bbox = self.image.getbbox()
        if bbox:
            cropped = self.image.crop(bbox)
        else:
            cropped = self.image
        resized = cropped.resize((28, 28))
        inverted = ImageOps.invert(resized)
        return inverted

    def run(self):
        self.window.mainloop()
