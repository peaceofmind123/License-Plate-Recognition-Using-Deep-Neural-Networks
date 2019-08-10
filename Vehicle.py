import re
import copy
class Vehicle:

    def __init__(self,id=0):
        self.id = id
        self.vehicle_imgs = []
        self.lp_imgs = []
        self.img_current = None
        self.license_number_predictions = []
        self.license_number:str = ""
        self.current_bounding_box_centroid = (0,0)
        self.bboxes = []
        self.bbox_current = None
        self.tokenized_lnums = []
        self.is_completely_processed = False

    def _is_number(self, num):
        try:
            int(num)
            return True
        except ValueError:
            return False


    def tokenize_lnums(self):
        tokenized_lnums = []

        for lnum in self.license_number_predictions:
            tokenized_lnum = lnum.split(' ')
            tokenized_lnums.append(tokenized_lnum)
        self.tokenized_lnums = tokenized_lnums
        return tokenized_lnums


    def aggregate_ocr(self):
        if len(self.license_number) > 0:
            return self.license_number
        self.clean_lnums()

        best_values = dict()
        for i, l_num in enumerate(self.tokenized_lnums):
            for j, symbol in enumerate(l_num):
                try:
                    best_values[j].append(symbol)
                except KeyError:
                    best_values[j] = []
                    best_values[j].append(symbol)

        for pos in best_values:
            count = dict()
            values_of_pos = best_values[pos]
            for char in values_of_pos:
                if char in count.keys():
                    count[char] += 1
                else:
                    count[char] = 1

            max_val = max(count.values())
            optimum_char = list(count.keys())[list(count.values()).index(max_val)]
            self.license_number = self.license_number + optimum_char

        matchObj = re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-z]+)([0-9]+)', self.license_number)
        try:
            self.license_number = matchObj.group()
        except:
            self.license_number = ''
        return self.license_number

    # initial cleaning code
    def clean_lnums(self):

        for i, lp_num in enumerate(self.license_number_predictions):
            flag = 0
            lp_num_no_str = "".join(lp_num.split())

            matchObj = re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-z]+)([0-9]+)', lp_num_no_str)
            if not matchObj:

                while True:
                    try:
                        self.license_number_predictions.remove(lp_num)
                    except Exception:
                        break

            else:
                if matchObj.group() != lp_num_no_str:
                    flag = 1
                    while True:
                        try:
                            self.license_number_predictions.remove(lp_num)
                        except Exception:
                            break


                first_part = matchObj.group(1)
                second_part = matchObj.group(2)
                third_part = matchObj.group(3)
                fourth_part = matchObj.group(4)
                if len(first_part) > 3 or len(second_part)>2 or len(third_part)>3 or len(fourth_part)>4:
                    while True:
                        try:
                            self.license_number_predictions.remove(lp_num)
                        except Exception:
                            break
        self.tokenize_lnums()
