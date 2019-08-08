
class Vehicle:

    def __init__(self,id=0):
        self.id = id
        self.vehicle_imgs = []
        self.img_current = None
        self.license_number_predictions = []
        self.license_number:str = ""
        self.current_bounding_box_centroid = (0,0)
        self.bboxes = []
        self.bbox_current = None

