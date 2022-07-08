import numpy as np
import cv2
from PIL import Image, ImageEnhance
import streamlit as st
from io import BytesIO
# print(cv2.__version__)
# https://medium.com/analytics-vidhya/meet-streamlit-sharing-build-a-simple-photo-editor-9d9e2e8872a
# https://www.analyticsvidhya.com/blog/2022/06/cartoonify-image-using-opencv-and-python/


@st.cache(allow_output_mutation=True)
def load():
    # EXTERNAL_DEPENDENCIES = {
    #     "caffe.caffemodel": {
    #         "url": "https://drive.google.com/file/d/1lDjVO8sGtZQ-LUqw2yFCWJatCVrJyFNf/view?usp=sharing",
    #         "size":  128946764}
    # }
    prototxt = "D:\\PROJECTS\\Newlify\\colorization_deploy_v2.prototxt"
    caffe_model = "D:\\PROJECTS\\Newlify\\colorization_release_v2.caffemodel"
    pts_npy = "pts_in_hull.npy"

    net = cv2.dnn.readNet(prototxt, caffe_model)
    pts = np.load(pts_npy)

    layer1 = net.getLayerId("class8_ab")
    # print(layer1)
    layer2 = net.getLayerId("conv8_313_rh")
    # print(layer2)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(layer1).blobs = [pts.astype("float32")]
    net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def newlyfy(image, net):
    test_image = image

    # test_image = cv2.imread(test_image)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

    normalized = test_image.astype("float32") / 255.0

    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab_image, (224, 224))

    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))

    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))

    L = cv2.split(lab_image)[0]

    LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)

    RGB_colored = np.clip(RGB_colored, 0, 1)

    RGB_colored = (255 * RGB_colored).astype("uint8")

    RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)

    return RGB_colored


def download(image):
    buf = BytesIO()
    PIL_image = Image.fromarray(
        np.uint8(image)).convert('RGB')
    PIL_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    # btn = st.download_button("Download", byte_im, 'Newlified')
    btn = st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="newlified.png",
        mime="image/jpeg",
    )


def sample():
    with st.sidebar.container():
        new_title = '<p style="font-family:Monospace; color:White; font-size: 25px;">Sample Images</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        img1 = Image.open("test\\barn.png")
        img2 = Image.open(
            "test\\WhatsApp Image 2022-03-14 at 09.24.36.jpeg")
        img3 = Image.open(
            "test\\Charlie-Chaplin-City-Lights.webp")
        img4 = Image.open(
            "test\\Pin by Chris Glad on Blackground....jpg")
        img5 = Image.open("test\\find-models-11.webp")
        img6 = Image.open(
            "test\\taj-mahal-1652183243.jpg")
        img7 = Image.open(
            "test\\5138683118fd41214d6d28ecb067d93e.jpg")
        st.image(img1, use_column_width=True)
        st.image(img2, use_column_width=True)
        st.image(img3, use_column_width=True)
        st.image(img5, use_column_width=True)
        st.image(img4, use_column_width=True)
        st.image(img6, use_column_width=True)
        st.image(img7, use_column_width=True)


def main():

    net = load()
    #st.sidebar.title("Developer's Contact")
    st.sidebar.markdown('[![Harsh-Dhamecha]'
                        '(https://img.shields.io/badge/Author-Sourav%20Dey-brightgreen)]'
                        '(https://github.com/souvenger)')
    Output_image = 400
    with st.sidebar.container():
        image = Image.open("Newlify.png")
        st.image(image, use_column_width=True)

    menu = ['Newlify', 'Purify', 'Blendify', 'Cartoonify']
    op = st.sidebar.selectbox('Play with OpenCV', menu)

    if(op == 'Newlify'):
        st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Newlify</p>',
                    unsafe_allow_html=True)

        st.text("\n")

        uploaded_image = st.file_uploader(
            "Upload Your Black and White Image ", type=['jpg', 'png', 'jpeg'])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.markdown('<p style="text-align: center;">Black and White</p>',
                            unsafe_allow_html=True)
                st.image(image, width=350)

            with col2:
                st.markdown('<p style="text-align: center;">Coloured</p>',
                            unsafe_allow_html=True)
                # print(uploaded_image.name)
                image = np.array(image.convert('RGB'))
                result = newlyfy(image, net)
                st.image(result, width=350)
                download(result)

    # Play with Opencv
    elif(op == "Purify"):
        st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Purify</p>',
                    unsafe_allow_html=True)

        img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if img is not None:

            image = Image.open(img)
            st.sidebar.text('Original Image')
            st.sidebar.image(image, use_column_width=True)
            result = image

            filters = st.sidebar.radio(
                'Filters', ['Grayscale', 'Sepia', 'Blur', 'Contour', 'Sketch'])
            if(filters == 'Grayscale'):
                img_convert = np.array(image.convert('RGB'))
                gray_image = cv2.cvtColor(img_convert, cv2.COLOR_RGB2GRAY)
                st.image(gray_image, width=Output_image)
                result = gray_image

            elif filters == 'Sepia':
                img_convert = np.array(image.convert('RGB'))
                img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
                kernel = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
                sepia_image = cv2.filter2D(img_convert, -1, kernel)
                st.image(sepia_image, channels='BGR', width=Output_image)
                result = sepia_image

            elif filters == 'Blur':
                img_convert = np.array(image.convert('RGB'))
                slide = st.sidebar.slider(
                    'blurryness', 1, 100, 9, step=2)
                img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
                blur_image = cv2.GaussianBlur(
                    img_convert, (slide, slide), 0, 0)
                st.image(blur_image, channels='BGR', width=Output_image)
                result = blur_image

            elif filters == 'Contour':
                img_convert = np.array(image.convert('RGB'))
                img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
                blur_image = cv2.GaussianBlur(img_convert, (11, 11), 0)
                canny_image = cv2.Canny(blur_image, 100, 150)
                st.image(canny_image, width=Output_image)
                result = canny_image

            elif filters == 'Sketch':
                img_convert = np.array(image.convert('RGB'))
                gray_image = cv2.cvtColor(img_convert, cv2.COLOR_RGB2GRAY)
                inv_gray = 255 - gray_image
                blur_image = cv2.GaussianBlur(inv_gray, (25, 25), 0, 0)
                sketch_image = cv2.divide(
                    gray_image, 255 - blur_image, scale=256)
                st.image(sketch_image, width=Output_image)
                result = sketch_image
            download(result)
    elif op == 'Blendify':
        st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Blendify</p>',
                    unsafe_allow_html=True)

        img1 = st.file_uploader("Upload First Image", type=[
                                'jpg', 'png', 'jpeg'])

        img2 = st.file_uploader("Upload Second Image",
                                type=['jpg', 'png', 'jpeg'])
        if img1 and img2:
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img1 = np.uint8(img1.convert('RGB'))
            img1 = cv2.resize(img1, (400, 400))
            img2 = np.uint8(img2.convert('RGB'))
            img2 = cv2.resize(img2, (400, 400))
            result = img1
            filters = st.sidebar.radio(
                'Select Opeartion', ['ADDITION', 'SUBTRACTION', 'BITWISE AND', 'BITWISE OR', 'BITWISE XOR'])

            if (filters == 'ADDITION'):
                wg1 = st.sidebar.slider(
                    'Weightage of First Image', 0.0, 1.0, 0.5, step=0.1)
                wg2 = st.sidebar.slider(
                    'Weightage of Second Image', 0.0, 1.0, 0.5, step=0.1)
                sum = cv2.addWeighted(img1, wg1, img2, wg2, 0)
                st.image(sum, width=Output_image)
                result = sum
            elif filters == 'SUBTRACTION':
                sum = cv2.subtract(img1, img2)
                st.image(sum, width=Output_image)
                result = sum
            elif filters == 'BITWISE AND':
                sum = cv2.bitwise_and(img1, img2)
                st.image(sum, width=Output_image)
                result = sum

            elif filters == 'BITWISE OR':
                sum = cv2.bitwise_or(img1, img2)
                st.image(sum, width=Output_image)
                result = sum

            elif filters == 'BITWISE XOR':
                sum = cv2.bitwise_xor(img1, img2)
                st.image(sum, width=Output_image)
                result = sum
            else:
                st.image(img1, width=Output_image)
                st.image(img2, width=Output_image)
                result = sum
            download(result)
    elif op == "Cartoonify":
        st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Cartoonify</p>',
                    unsafe_allow_html=True)

        img = st.file_uploader("Upload Image", type=[
            'jpg', 'png', 'jpeg'])
        if img is not None:
            img = Image.open(img)
            img = np.array(img.convert('RGB'))
            ReSized1 = cv2.resize(img, (200, 200))
            grayScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ReSized2 = cv2.resize(grayScaleImage, (200, 200))
            smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
            ReSized3 = cv2.resize(smoothGrayScale, (200, 200))
            getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 9, 9)
            ReSized4 = cv2.resize(getEdge, (200, 200))
            # and keep edge sharp as required
            colorImage = cv2.bilateralFilter(img, 9, 300, 300)
            ReSized5 = cv2.resize(colorImage, (200, 200))
            # masking edged image with our "BEAUTIFY" image
            cartoonImage = cv2.bitwise_and(
                colorImage, colorImage, mask=getEdge)
            ReSized6 = cv2.resize(cartoonImage, (200, 200))

            st.image(cartoonImage)
            download(cartoonImage)
            st.header("\nSteps for Cartoonify")
            st.image([ReSized1, ReSized2, ReSized3, ReSized4, ReSized5,
                     ReSized6], caption=['Original', 'Grayscale Image', 'SmoothGrayscale Image', 'Edged Image', 'Mask Image', 'Cartoon Image'])

            # st.image(ReSized2, caption='Grayscale Image')
            # st.image(ReSized3, caption='SmoothGrayscale Image')
            # st.image(ReSized4, caption='Edged Image')
            # st.image(ReSized5, caption='Mask Image')
            # st.image(ReSized6, caption='Cartoon Image')
    st.text("\n")
    sample()


if __name__ == "__main__":
    main()
