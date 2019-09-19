from datetime import datetime
import time
from flask import Flask, render_template
from . import app

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")

@app.route("/aigo_prime")
def aigo_prime():
    return render_template("aigo_prime.html")

# image stylization
from image_stylization.ImageStylization import ImageStylization, ImageStylizationOptions
from image_stylization import image_utils

checkpoint = 'static/image_stylization/checkpoints/model.ckpt'
output_dir = 'static/image_stylization/images/stylized_images'
image_stylization_options = ImageStylizationOptions(checkpoint, output_dir)
print('ImageStylizationOptions \n'+image_stylization_options.to_string())
global image_stylization
image_stylization = ImageStylization(image_stylization_options)

@app.route("/stylization")
def stylization():
    global image_stylization

    # stylize image
    start_time = time.time()
    style_image_path = 'static/image_stylization/images/style_images/qingming_festival.jpg'
    content_image_path = 'static/image_stylization/images/content_images/taipei_101.jpg'
    style_image = image_utils.load_np_image_uint8(style_image_path)[:, :, :3]
    content_image = image_utils.load_np_image_uint8(content_image_path)[:, :, :3]
    stylized_image, stylized_image_path = image_stylization.stylize_image(style_image, content_image)
    end_time = time.time()
    print('stylize image cost {:.2f} ms'.format((end_time-start_time)*1000))
    print('stylized_image_path: '+stylized_image_path)

    # # stylize images by path
    # start_time = time.time()
    # style_images_paths='static/image_stylization/images/style_images/*.jpg'
    # content_images_paths='static/image_stylization/images/content_images/*.jpg'
    # stylized_images_path = image_stylization.stylize_images_by_path(style_images_paths, content_images_paths)
    # end_time = time.time()
    # print('stylize images by path cost {:.2f} ms'.format((end_time-start_time)*1000))
    # print('stylized_images_path: ')
    # print(stylized_images_path)

    return render_template("stylization.html", stylized_image=stylized_image_path
        , style_image=style_image_path, content_image=content_image_path)



    