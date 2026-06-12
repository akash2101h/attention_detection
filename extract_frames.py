import cv2, os

videos = [
    'C:/Users/agk21/OneDrive/Desktop/classroom/video1.mp4',
    'C:/Users/agk21/OneDrive/Desktop/classroom/video2.mp4',
    'C:/Users/agk21/OneDrive/Desktop/classroom/video3.mp4',
    'C:/Users/agk21/OneDrive/Desktop/classroom/video4.mp4',
]

output_folder = 'C:/Users/agk21/OneDrive/Desktop/classroom/images'
os.makedirs(output_folder, exist_ok=True)

total_saved = 0
for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    count = 0
    vname = os.path.basename(video_path).replace('.mp4', '')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            cv2.imwrite(f'{output_folder}/{vname}_frame_{count:04d}.jpg', frame)
            total_saved += 1
        count += 1
    cap.release()
    print(f'{vname}: done')

print(f'Total frames saved: {total_saved}')