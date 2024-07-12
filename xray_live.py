################################################################################
# Example : Perform live X-ray inference on image, video, webcam input
# using different object tracking algorithms.
# Copyright (c) 2022 - Neelanjan Bhowmik / Yona Falinie / Toby Breckon 
# Durham University, UK
# License : 
################################################################################

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
from mmcv.image import imread
import torch
from tools import camera_stream

import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import seaborn as sns
import os
import numpy as np
import json
import math
# import imutils
import time
from tabulate import tabulate
import shutil
################################################################################

def draw_bbox_pil(args, im, img, bbox_result, segm_result, labels, colors):
    # print('bbox_result ', bbox_result)
    # font_size = int((16.0/640.0)*src_img.size[0])
    font = ImageFont.truetype("arial", 12)
    
    bboxes = np.vstack(bbox_result)
    img_cp = img.copy()
    h, w, c = img.shape
    blank_image = np.zeros((h,w,c), np.uint8)
    blank_image.fill(255)
    ##seg draw
    if segm_result is not None:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = (np.where(bboxes[:, -2] > args.score_thr)[0])

        indices_score_thr = np.where(bboxes[:, -2] > args.score_thr)[0]
        indices_anomaly_threshold = np.where(bboxes[:, -1] < args.anomaly_threshold)[0]

        set_indices_score_thr = set(indices_score_thr)
        set_indices_anomaly_threshold = set(indices_anomaly_threshold)

        inds = set_indices_score_thr & set_indices_anomaly_threshold
        inds = list(inds)
        # inds_array = np.array(list(inds))

        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        inds_mask = inds
        label_mask = labels

        # for i in inds:
        #     i = int(i)
        #     print('-', i)
        #     color_mask = color_masks[labels[i]]
        #     color_mask[0][0] = 0
        #     color_mask[0][1] = 0
        #     color_mask[0][2] = 0
        #     mask = segms[i].astype(bool)
        #     img[mask] = img[mask] * 256 + color_mask * 1

    ##bbox draw 
    # assert bboxes.ndim == 2
    # assert labels.ndim == 1
    # assert bboxes.shape[0] == labels.shape[0]
    # assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    # img = imread(img)

    if args.score_thr > 0:
        # assert bboxes.shape[1] == 5
        # print(bboxes)
        scores = bboxes[:, -2]
        anomaly_scores = bboxes[:, -1]
        # print('scores-->', scores)
        # print('scores-->', anomaly_scores)
        # inds = scores > args.score_thr
        # inds = anomaly_scores > args.anomaly_threshold and scores > args.score_thr
        # scores = np.array(scores)
        # anomaly_scores = np.array(anomaly_scores)

        # inds = np.bitwise_and(anomaly_scores < args.anomaly_threshold, scores > args.score_thr)
        
        inds = (anomaly_scores < args.anomaly_threshold) & (scores > args.score_thr)
        # print(inds)

        bboxes = bboxes[inds, :]
        scores_f = scores[inds]
        anomaly_scores_f = anomaly_scores[inds] 
        # labels = labels[inds]
        # print('bboxes-->', bboxes)
        # print('score-->', scores)
    
    img = np.ascontiguousarray(img)

    img_mask_list = []

    for j, (bbox, bscore, ascore) in enumerate(zip(bboxes, scores_f, anomaly_scores_f)):
        bbox_int = bbox.astype(np.int32)
        # print('in loop: ', bbox_int, bscore, ascore)
        
        # label_text = CLASSES[
        #     label] if CLASSES is not None else f'cls {label}'

        color = colors[1]
        color = tuple([int(x*255) for x in color])
        
        if segm_result is not None:
            # print('mask')
            img_cp_mask = img_cp.copy()
            i = int(inds_mask[j])
            color_mask = color_masks[label_mask[i]]
            # 29, 177, 207
            # BGR
            color_mask[0][0] = 0
            color_mask[0][1] = 0
            color_mask[0][2] = 255
            # mask = np.logical_not(segms)[i].astype(bool)
                                            
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5    
            # img[mask] = 255
        # print('no mask')
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        
        # cv2.rectangle(
        #     img, left_top, right_bottom, color=color, thickness=1)
        # if len(bbox) > 4:
        #     label_text += f'|{bbox[-1]:.02f}'

        # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
        #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color)
        
        # Convert to PIL
        if len(bbox) > 4:
            # print('PIL')
            label_text = f'|{bscore:.02f}||{ascore:.02f}'
            src_img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(src_img)
            draw.rectangle([bbox_int[0],bbox_int[1],bbox_int[2],bbox_int[3]], outline=tuple(color), width=4)
            text_size = font.getsize(label_text)
            button_size = (text_size[0], text_size[1]+2)
            button_img = Image.new('RGBA', button_size, "black")
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((0, 0), label_text, font=font, fill=tuple(color))
            src_img.paste(button_img, (int(bbox_int[0]),int(bbox_int[1])), mask=Image.new("L", button_img.size, 200))
            img=np.array(src_img)  
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# else:
#     print('Invalid target class.')
    
    
    # if args.crop == 'yes':
    #     if segm_result is not None and len(img_mask_list) > 0:
    #         os.makedirs(f'{args.output}/seg',exist_ok=True)
    #         img_mask_final = img_mask_list[0]
    #         for i in img_mask_list:
    #             img_mask_final = cv2.bitwise_and(img_mask_final, i)
    #         cv2.imwrite(f'{args.output}/seg/{im}', img_mask_final)

    #     else:
    #         cv2.imwrite(f'{args.output}/seg/{im}', blank_image)

    return img
################################################################################

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('--input', help='Input image directory/image')
    parser.add_argument('--video', help='Input video path')
    parser.add_argument('--webcam', action="store_true", help="Take inputs from webcam")
    parser.add_argument('--output', help='Output image directory')
    parser.add_argument('--db', type=str, help='dataset name')
    parser.add_argument('--yes_cls', type=str, help='Traget object class, - separated [car-dog], defuault all classes.')
    parser.add_argument('--dbpath', default='../dataset', type=str, help='dataset directory path')
    parser.add_argument('--dataroot', default='', type=str, help='directory path to perform live demo')
    parser.add_argument('--anomaly-threshold', type=float, default=0.995)
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: png)',
        default='png',
        type=str
    )
    parser.add_argument(
        "--camera_to_use",
        type=int,
        default=0,
        help="Specify camera to use for webcam option")
    parser.add_argument(
        '--endswith', 
        help='Demo images that endswith certain str', 
        default='color.png', )

    parser.add_argument('--crop', default='no', type=str, help='Crop detected objets [yes]')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        "-fs",
        "--fullscreen",
        action='store_true',
        help="run in full screen mode")
    args = parser.parse_args()

    t_val = []
    for arg in vars(args):
        t_val.append([arg, getattr(args, arg)])
    print(tabulate(t_val, 
        ['input', 'value'], 
        tablefmt="psql")) 

    WINDOW_NAME = 'Detection'
    CLASSES = []
    # with open(f'{args.dbpath}/{args.db}/annotation/{args.db}_train.json') as f:
    #    json_data = json.load(f)
    # for data_id, data_info in json_data.items():
    #     if data_id == 'categories':
    #         for cat in data_info:
    #             CLASSES.append(cat['name'])
    # # CLASSES = tuple(CLASSES)
    # colors = sns.color_palette("husl", len(CLASSES))
    colors = sns.color_palette("husl", 2)
    # print('CLASSES ', CLASSES)
    bbox_thrs = 0
    # if args.yes_cls is not None:
    #     args.yes_cls = args.yes_cls.split("-")
    # else:
    #     args.yes_cls = CLASSES

    # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, 
    #     tuple(CLASSES), args.checkpoint, 
    #     device=args.device)
    
    model = init_detector(args.config,
        args.checkpoint, device=args.device)
   
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    fps = []
    infer_sec = []
    tmp_path = f'output/tmp'
    os.makedirs(tmp_path, exist_ok=True)
    fcnt = 1 

    fps_all = []
    infer_sec = []

    ####Live demo
    if args.dataroot:
        KNOWN_FILES = [i for i in os.listdir(args.dataroot) if i.endswith(args.endswith)]
        img_hist = []

        if not(args.output):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)    
        # Live to X-ray machine server
        print('\n|__Listening')
        while True:
            t1 = time.time()
            filenames = [f for f in os.listdir(args.dataroot) if f.endswith(args.endswith)]
        
            for filename in filenames:
                if filename not in KNOWN_FILES:
                    print("|__File added: {}".format(filename))
                    t2 = time.time()
                    result = inference_detector(model, f'{args.dataroot}/{filename}')    
                    t3 = time.time()
                    inference_time = t3 - t2
                    fps_frame = 1.0/inference_time
                    fps.append(fps_frame)
                    infer_sec.append(inference_time)
                    print(f'\t\tFPS: {fps_frame}, Sec: {inference_time}')

                    img = mmcv.imread(f'{args.dataroot}/{filename}')

                    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    # cv2.imshow(WINDOW_NAME, img)
                    # if cv2.waitKey(0) == 27:
                    #     exit()


                    # img = img.copy()
                    if isinstance(result, tuple):
                        bbox_result, segm_result = result
                        if isinstance(segm_result, tuple):
                            segm_result = segm_result[0]  # ms rcnn
                    else:
                        bbox_result, segm_result = result, None
                    # bboxes = np.vstack(bbox_result)
                    # print('bbox_result', bbox_result)
                    labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)

                    img = draw_bbox_pil(
                        args,
                        filename, 
                        img, 
                        bbox_result, 
                        segm_result, 
                        labels,
                        colors
                    )

                    if args.output and args.crop != 'yes':
                        cv2.imwrite(f'{args.output}/{filename}',img)
                    else:
                        # cv2.imwrite(f'{tmp_path}/{fcnt}_{filename}',img)
                        # cv2.imwrite(f'{tmp_path}/{filename}',img)
                        # img_hist.append(f'{tmp_path}/{filename}')
                        
                        cv2.imshow(WINDOW_NAME, img)  
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN & args.fullscreen)  

                        key = cv2.waitKey(0) & 0xFF
                    
                        # stop_tik = ((cv2.getTickCount() - start_tik) /
                        #             cv2.getTickFrequency()) * 1000
                                
                        # key = cv2.waitKey(
                        #     max(2, 40 - int(math.ceil(stop_tik)))) & 0xFF

                        # press "x" for exit  / press "f" for fullscreen
                        if (key == ord('x')):
                            shutil.rmtree(tmp_path)
                            exit()
                            
                        elif (key == ord('f')):
                            args.fullscreen = not(args.fullscreen)    

                        # elif (key == ord('b')):
                        #     wd2 = 'Previous Detection'
                        #     img_display = img_hist[-2]

                        #     print('|__Currently on display: ', os.path.basename(img_display))
                        #     if os.path.exists(img_display):
                        #         img_d = mmcv.imread(img_display)
                        #         cv2.namedWindow(wd2, cv2.WINDOW_NORMAL)
                        #         cv2.imshow(wd2, img_d)
                        #         if cv2.waitKey(5):
                        #             break
                        # # elif (key == ord('n')):
                        # #     img_display = f'{fcnt-1}_{filename}'
                        # #     if os.path.exists(f'{tmp_path}/{img_display}'):
                        # #         cv2.imshow(WINDOW_NAME, f'{tmp_path}/{img_display}')            

                        # fcnt+=1
                        # 3_gilardoni-fep-me640amx-20180305-170900-00004174-service-02-color
                        # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        # cv2.imshow(WINDOW_NAME, img)
                        # if cv2.waitKey(0) == 27:
                        #     exit()
                
                KNOWN_FILES.append(filename)

    
    ####Image Directory
    if args.input:
        if os.path.isdir(args.input):
            for im in sorted(os.listdir(args.input)):
                print('|__Image: ', im)
                
                start_time = time.time()
                result = inference_detector(model, f'{args.input}/{im}')
                end_time = time.time()
                inference_time = end_time - start_time
                fps_frame = 1.0/inference_time
                fps.append(fps_frame)
                infer_sec.append(inference_time)
                print(f'\t\tFPS: {fps_frame}, Sec: {inference_time}')
                # show_result_pyplot(model, f'{args.input}/{im}', args.output, result, score_thr=args.score_thr)

                ####
                
                img = mmcv.imread(f'{args.input}/{im}')

                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = result, None
                bboxes = np.vstack(bbox_result)
                # print('bbox_result', bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                
                img = draw_bbox_pil(
                    args,
                    im, 
                    img, 
                    bbox_result, 
                    segm_result, 
                    labels,
                    colors
                )

                if args.output and args.crop != 'yes':
                    cv2.imwrite(f'{args.output}/{im}',img)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, img)
                    if cv2.waitKey(0) == 27:
                        exit()
            avg_fps = sum(fps)/len(fps)
            avg_sec = sum(infer_sec)/len(infer_sec)
            print(f'\n|__Average fps {int(avg_fps)}')
            print(f'|__Average ms {(avg_sec*1000)}')
        ####Single Image
        else:
            im = args.input
            print('Image: ', im)
            result = inference_detector(model, im)
            ####
            img = mmcv.imread(im)

            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, img)
            # if cv2.waitKey(0) == 27:
            #     exit()

            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            img = draw_bbox_pil(
                args,
                im, 
                img, 
                bbox_result, 
                segm_result, 
                labels,
                colors
            )

            if args.output and args.crop != 'yes':
                cv2.imwrite(f'{args.output}/{im}',img)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, img)
                if cv2.waitKey(0) == 27:
                    exit()
        
    if args.video or args.webcam:

        # define video capture object
        try:
            # to use a non-buffered camera stream (via a separate thread)
            if not(args.video):
                cap = camera_stream.CameraVideoStream()
            else:
                cap = cv2.VideoCapture()  # not needed for video files

        except BaseException:
            # if not then just use OpenCV default
            print("INFO: camera_stream class not found - camera input may be buffered")
            cap = cv2.VideoCapture()

        
        if args.output:
            os.makedirs(f'{args.output}/video', exist_ok=True)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        if args.video:
            if os.path.isdir(args.video):
                lst_vid = [os.path.join(args.video, file)
                        for file in os.listdir(args.video)]
            if os.path.isfile(args.video):
                lst_vid = [args.video]
        if args.webcam:
            lst_vid = [args.camera_to_use]

        for vid in lst_vid:
            keepProcessing = True
            if args.video:
                print(f'\n|__Processing video input: {vid}')
            if args.webcam:
                print('\n|__Processing webcam input >>')

            if cap.open(vid):
                # get video information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if args.output and args.video:
                    f_name = os.path.basename(vid)
                    out = cv2.VideoWriter(
                        filename=f'{args.output}/video/{f_name}',
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                        fps=float(fps),
                        frameSize=(width, height),
                        isColor=True,
                    )

                while (keepProcessing):
                    start_t = time.time()
                    # start a timer (to see how long processing and display takes)
                    start_tik = cv2.getTickCount()

                    # if camera/video file successfully open then read frame
                    if (cap.isOpened):
                        ret, frame = cap.read()
                        # when we reach the end of the video (file) exit cleanly
                        if (ret == 0):
                            keepProcessing = False
                            continue

                    result = inference_detector(model, frame)

                    stop_t = time.time()
                    inference_time = stop_t - start_t
                    fps_frame = 1.0/inference_time
                    fps_all.append(fps_frame)
                    infer_sec.append(inference_time)
                    print(f'\t\tFPS: {round(fps_frame,3)}, Sec: {round(inference_time,3)}')

                    if isinstance(result, tuple):
                        bbox_result, segm_result = result
                        if isinstance(segm_result, tuple):
                            segm_result = segm_result[0]  # ms rcnn
                    else:
                        bbox_result, segm_result = result, None
                    
                    labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)

                    frame = draw_bbox_pil(
                        args,
                        frame, 
                        frame, 
                        bbox_result, 
                        segm_result, 
                        labels,
                        colors
                    )

                    if args.output and args.video:
                        out.write(frame)
                    
                    else:
                        cv2.imshow(WINDOW_NAME, frame)
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN & args.fullscreen)

                        stop_tik = ((cv2.getTickCount() - start_tik) /
                                    cv2.getTickFrequency()) * 1000
                        key = cv2.waitKey(
                            max(2, 40 - int(math.ceil(stop_tik)))) & 0xFF

                        # press "x" for exit  / press "f" for fullscreen
                        if (key == ord('x')):
                            keepProcessing = False
                        elif (key == ord('f')):
                            args.fullscreen = not(args.fullscreen)

            if args.output and args.video:
                out.release()
            else:
                cv2.destroyAllWindows()

        avg_fps = sum(fps_all)/len(fps_all)
        avg_sec = sum(infer_sec)/len(infer_sec)
        print(f'\n|__Average fps {int(avg_fps)}')
        print(f'|__Average ms {round((avg_sec*1000),3)}')

if __name__ == '__main__':
    main()
