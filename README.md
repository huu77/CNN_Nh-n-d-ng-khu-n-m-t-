# CNN_Nhận diện khuôn mặt
bạn tạo forder mới xong thì thay đổi cái này 
label_dict = {'huu': [1,0], 'mask': [0,1]}
# lưu ý :
neeus them label thi tang so luong len der ko loi , vd 3 nguoi thi 100, 010, 001  , vd 4 người  4 bit , n người thì n bit

# các bước run :
- chạy file addimage.py trước , nhập tên của bạn , chụp 100 bức ảnh , nếu muốn chụp theo ý thì chỉ cần chỉnh ####numberFace = n ###là đc
- chạy file module trainModule.py để train models .
- chạy file useCam.py để test .
- # lưu ý : nếu muốn chạy test video thì để chế độ này  'cam = cv2.VideoCapture('video.mp4')' với name video bạn tạo sẵn
- #         nếu bạn muốn sử dụng cammera thì cam = cv2.VideoCapture(0)

- Code trên dùng để học tập , và dùng file convolutions.py để giải thích 1 số layer để chúng ta hiểu sâu về thuật toán này hơn. Còn phần layer
model_training_first.add(layers.Flatten())
model_training_first.add(layers.Dense(1000, activation='relu'))
model_training_first.add(layers.Dense(256, activation='relu'))
model_training_first.add(layers.Dense(2, activation='softmax'))
-thì phần này chúng ta không có khả năng tạo :) , mọi người nên tìm hiểu thêm .
# Bạn có thể tham khảo bài đọc này : https://nttuan8.com/bai-6-convolutional-neural-network/
