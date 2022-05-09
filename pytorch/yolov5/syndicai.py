import torch


class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        self.model = torch.load('Lodel.pt').autoshape()

    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        y = self.model(payload["sample"])

        # # Draw boxes
        # boxes = results.xyxy[0].numpy()
        # box_img = draw_box(img, boxes)

        # # Save image
        # #box_img.save("sample_data/output.png", "PNG")

        return y