import tkinter as tk
import Yolo_high as yolo
import localizer_high as localizer
import ocr_api as ocr
import sys
import cv2
import numpy
import os
import object_tracker_api as obj_tracker
from Vehicle import Vehicle
import PIL.Image, PIL.ImageTk

VIDEO_SOURCE = '1.MOV'
WINDOW_TITLE = 'License Plate Recognition'
FRAME_DELAY = 15

# global vars to be used
frame_no = 0
id_for_new_vehicle = 0
# initialization
yolo.initialize_model()
localizer.initialize_model()
tracked_vehicles_info = {}
tracked_vehicle_ids = []
current_tracked_id = 0
tracked_vehicles = [] # stores the actual vehicle object
vehicle = None

class App:
    def __init__(self, window, window_title='WINDOW_TITLE',video_source=VIDEO_SOURCE):
        self.window = window
        self.window_title = window_title
        self.window.geometry('1920x1080')
        self.delay = FRAME_DELAY

        # open video
        self.vid = MyVideoCapture(os.path.join(os.getcwd(),'video',video_source))

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=1080, height=700)
        self.canvas.pack(side=tk.LEFT, fill=tk.Y)
        self.info_frame = tk.Frame(window,width=1920-1080, height=700)
        self.label_vehicle_no = tk.Entry(self.info_frame, width=50)
        self.vehicle_no = tk.Entry(self.info_frame, width=50)
        self.vehicle_lp = tk.Entry(self.info_frame, width=50)
        self.label_vehicle_lp = tk.Entry(self.info_frame, width=50)

        self.label_vehicle_no.insert(0, 'Vehicle No.')
        self.label_vehicle_lp.insert(0, 'Vehicle License Plate')

        self.label_vehicle_no.config(state="readonly")
        self.label_vehicle_lp.config(state="readonly")
        self.vehicle_no.config(state="readonly")
        self.vehicle_lp.config(state="readonly")

        self.label_vehicle_no.grid(row=0, column=0,pady=5)
        self.label_vehicle_lp.grid(row=0,column=1,pady=5)
        self.vehicle_no.grid(row=1,column=0)
        self.vehicle_lp.grid(row=1,column=1)


        self.info_frame.pack(side=tk.LEFT,fill=tk.Y)

        self.update()
        self.window.mainloop()

    def update(self):
        # TODO: add core code here
        global frame_no, id_for_new_vehicle, tracked_vehicles_info, tracked_vehicle_ids, current_tracked_id,tracked_vehicles,vehicle
        flag = 0
        ret, frame = self.vid.get_frame()
        # if frame_no % 1 != 0:
        #     frame_no += 1
        #     frame = cv2.resize(frame, (1080, 700))
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        #     self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        #     self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        #     return self.window.after(self.delay, self.update)

        if ret:
            vehicle_ids_this_frame = []
            ocr_outputs_this_frame = []
            # perform object detection
            vehicle_imgs, vbbox, bbox = yolo.get_vehicle_imgs(frame, 750, 700)
            if len(vehicle_imgs) == 0 or len(vbbox) == 0 or len(bbox) == 0:
                frame_no +=1
                frame = cv2.resize(frame, (1080, 700))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                return self.window.after(self.delay,self.update)
            # perform object tracking
            tracked_vehicles_info, id_for_new_vehicle = obj_tracker.track_vehicle(
                obj_tracker.get_bbox_without_class(bbox))

            # create or update the vehicles tracked till now
            for key in tracked_vehicles_info.keys():
                _, centroid_x, centroid_y = tracked_vehicles_info[key]
                try:
                    v_index = list(map(lambda obj: obj.id, tracked_vehicles)).index(key)
                    # vehicle is present in the current list of vehicles
                    vehicle = tracked_vehicles[v_index]

                except ValueError as e:  # vehicle is not found
                    if vehicle is not None:  # here vehicle refers to the last detected vehicle
                        vehicle.aggregate_ocr()
                        print(vehicle.license_number)
                    vehicle = Vehicle(key)
                    tracked_vehicles.append(vehicle)

                vehicle_ids_this_frame.append(vehicle.id)
                vehicle.current_bounding_box_centroid = (centroid_x, centroid_y)

                # find the vehicle's image in the current frame
                for v_img, v_bbox in zip(vehicle_imgs, vbbox):
                    # vehicle_imgs and vbbox have one-to-one direct correspondence

                    if obj_tracker.check_exact_match(vehicle.current_bounding_box_centroid[0],
                                                     vehicle.current_bounding_box_centroid[1],
                                                     v_bbox):
                        vehicle.img_current = v_img
                        vehicle.bbox_current = v_bbox
                        vehicle.bboxes.append(v_bbox)
                        vehicle.vehicle_imgs.append(v_img)

                # if vehicle has no corresponding image, no point to process it further
                if vehicle.img_current is None or vehicle.bbox_current is None:
                    tracked_vehicles.remove(vehicle)

            if len(tracked_vehicles) == 0:  # if no vehicle was detected, just skip
                frame_no += 1
                frame = cv2.resize(frame, (1080, 700))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                return self.window.after(self.delay,self.update)


            for v in tracked_vehicles:
                # perform lp localization
                if v.id in vehicle_ids_this_frame:
                    lp_imgs, lp_bboxes, frame = localizer.predict_license_plate([v.img_current], frame,
                                                                                [v.bbox_current])
                    lp_img = lp_imgs[0]
                    lp_img = cv2.cvtColor(lp_img, cv2.COLOR_RGB2GRAY)
                    ocr_output = ocr.predict_ocr(lp_img)
                    v.license_number_predictions.append(ocr_output if ocr_output is not None else '')
                    ocr_outputs_this_frame.append(ocr_output)

            # processing done, time for outputs
            # print(frame_no, ocr_outputs_this_frame)

            # create the frame to display
            frame = yolo.draw_bbox(frame, bbox)
            for v in tracked_vehicles:
                if v.id in vehicle_ids_this_frame:
                    cx = int(v.current_bounding_box_centroid[0])
                    cy = int(v.current_bounding_box_centroid[1])
                    frame = cv2.putText(frame, f'{v.id}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 4)
            # resize the frame for display
            frame = cv2.resize(frame, (1080, 700))
            # cv2.imshow("prediction", frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            frame_no += 1

        self.window.after(self.delay,self.update)
class MyVideoCapture:
    def __init__(self, video_source =VIDEO_SOURCE):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError('Unable to open video',video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, frame
            else:
                return ret, None
        else:
            return None, None

App(tk.Tk(),'License Plate Recognition')