import tensorflow as tf
import os

# Helper Functions
def crop_image(image: tf.Tensor) -> tf.Tensor:
    """Returns a square image."""

    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = (shape[1] - new_shape) // 2
    offset_x = (shape[2] - new_shape) // 2

    image = tf.image.crop_to_bounding_box(
        image=image,
        offset_height=offset_y, offset_width=offset_x,
        target_height=new_shape, target_width=new_shape
    )

    return image


def load_image(image_url:str, image_size:tuple=(256,256), preserve_aspect_ratio:bool=True) -> tf.Tensor:
    # Cache the file locally.
    MAX_FILE_NAME_LEN = 128 # 128 characters max.

    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-MAX_FILE_NAME_LEN:], image_url, cache_dir=".")

    # Load and convert to float32 tensor, add batch dimension, and normalize to [0, 1]
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32
    )
    img = img[tf.newaxis, ...]

    img = crop_image(img)
    img = tf.image.resize(img, size=image_size, preserve_aspect_ratio=True)

    return img

def vgg_layers(layer_names:list[str]) -> tf.keras.models.Model:
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")

    for layer in vgg.layers:
        layer.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.models.Model(inputs=[vgg.input], outputs=outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers:list[str], content_layers:list[str], **kwargs):
        super().__init__(**kwargs)

        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.n_style_layers = len(style_layers)

        self.vgg.trainable = False
    
    def call(self, inputs:tf.Tensor) -> dict[str,dict[str,tf.Tensor]]:
        # Expects an input between [0,1]
        inputs = inputs * 255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)

        style_outputs, content_outputs = (
            outputs[:self.n_style_layers], outputs[self.n_style_layers:]
        )

        style_repr = [self._gram_matrix(style_out) for style_out in style_outputs]

        content_dict = {
            content_layer_name: content_out
            for content_layer_name, content_out in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_layer_name: style_out
            for style_layer_name, style_out in zip(self.style_layers, style_repr)
        }

        output_dict = {
            "content": content_dict,
            "style": style_dict
        }

        return output_dict
    
    def _gram_matrix(self, input_tensor:tf.Tensor) -> tf.Tensor:
        result = tf.linalg.einsum("bkpi, bkpj->bij", input_tensor, input_tensor)
        return result

class StylizeImage(tf.keras.models.Model):
    STYLE_LAYERS = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
    ]

    CONTENT_LAYERS = [
        "block4_conv2"
    ]

    def __init__(self, style_img:tf.Tensor, style_weight:float=1, content_weight:float=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.style_img = style_img
        self._extractor = StyleContentModel(StylizeImage.STYLE_LAYERS, StylizeImage.CONTENT_LAYERS)

        self._style_targets = self._extractor(self.style_img)['style']

        self.style_weight = style_weight
        self.content_weight = content_weight

        self._optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-2)
    
    def style(self, content_img:tf.Tensor, epochs:int=10, steps_per_epoch:int=10):
        output_img = tf.Variable(content_img)
        content_targets = self._extractor(content_img)['content']

        step = 0
        for epoch_idx in range(epochs):
            for step_idx in range(steps_per_epoch):
                step += 1
                self._train_step(output_img, content_targets)
                print(".", end="", flush=True)
            print(f"Train step: {step}.")
        
        return output_img

    @tf.function
    def _train_step(self, output_img:tf.Tensor, content_targets:tf.Tensor):
        with tf.GradientTape() as tape:
            outputs = self._extractor(output_img)
            loss = self._overall_loss(outputs, content_targets)

        grad = tape.gradient(loss, output_img)
        self._optimizer.apply_gradients([(grad, output_img)])
        output_img.assign(self._clip_0_1(output_img))


    def _mse_loss(self, preds:dict[str,tf.Tensor], targets:dict[str,tf.Tensor]) -> float:
        mses = [tf.reduce_mean((preds[name] - targets[name]) ** 2) for name in preds.keys()]
        return tf.reduce_mean(mses)

    def _overall_loss(self, outputs:dict[str,dict[str,tf.Tensor]], content_targets:tf.Tensor) -> float:
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        style_loss = self.style_weight * self._mse_loss(style_outputs, self._style_targets)
        content_loss = self.content_weight * self._mse_loss(content_outputs, content_targets)

        return style_loss + content_loss

    def _clip_0_1(self, img:tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'
    style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
    
    output_img_size = (384, 384)
    style_img_size = (256, 256)
    n_epochs = 10
    n_steps_per_epoch = 5

    content_img = load_image(content_image_url, output_img_size)
    style_img = load_image(style_image_url, style_img_size)

    styler = StylizeImage(style_img)

    output = styler.style(content_img, epochs=n_epochs, steps_per_epoch=n_steps_per_epoch)


    plt.imshow(tf.squeeze(output, axis=0))
    plt.show()

