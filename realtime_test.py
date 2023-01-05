from pypylon import pylon
import cv2
import argparse
import numpy as np
import time
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        image = image.GetArray()

        # Đọc các thông số của ảnh
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        classes = None
        ap = argparse.ArgumentParser()
        ap.add_argument('-i', '--image', required=True,
                        help='path to input image')
        ap.add_argument('-c', '--config', required=True,
                        help='path to yolo config file')
        ap.add_argument('-w', '--weights', required=True,
                        help='path to yolo pre-trained weights')
        ap.add_argument('-cl', '--classes', required=True,
                        help='path to text file containing class names')
        args = ap.parse_args()


        def get_output_layers(net):
            layer_names = net.getLayerNames()

            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers


        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            label = str(classes[class_id])
            confidence = str(round(confidence * 100, 2))

            color = COLORS[class_id]

            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

            cv2.putText(img, confidence, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img, label, (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        # Đưa ảnh vào mạng và đặt thông số để xuất các bouding box
        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5  # Đây là ngưỡng vật thể, nếu xác suất của vật thể nhỏ hơn 0.5 thì #model sẽ loại bỏ vật thể đó.
        nms_threshold = 0.4  # Nếu có nhiều box chồng lên nhau, và vượt quá giá trị 0.4 ( tổng diện tích chồng nhau) thì 1 trong 2 box sẽ bị loại bỏ

        # Thực hiện xác định bằng HOG và SVM
        start = time.time()  # Đo thời gian thực thi của model:tính FPS ( số lượng ảnh xử lý được trong 1 giây)
        # Để tìm cách tối ưu hoặc tính toán phù hợp cho các bài toán thời gian thực khác nhau.

        # Dự đoán vật thể trên các ô lưới và lưu giá trị vào 1 mảng:
        for out in outs:
            for detection in out:
                # Trích xuất điểm số, phân loại và độ tin cậy của dự đoán
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Chỉ xem xét các dự đoán cao hơn ngưỡng tin cậy
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    # Sử dụng tọa độ trung tâm, chiều rộng và chiều cao để lấy tọa độ của góc trên cùng bên trái
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # Lấy ngưỡng các dự đoán
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)  # Apply Non-Max Suppression
        # Lưu giá trị các bounding box và vẽ các bounding box lên hình
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        cv2.imshow("output", image)
        # Kết thúc đo thời gian thực thi và in thời gian thực thi lên màn hình
        end = time.time()
        print("YOLO Execution time: " + str((end - start)))

        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()