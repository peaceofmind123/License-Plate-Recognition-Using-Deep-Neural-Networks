
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
        self.tokenized_lnums = []

    def tokenize_lnums(self):
        tokenized_lnums = []

        for lnum in self.license_number_predictions:
            tokenized_lnum = lnum.split(' ')
            tokenized_lnums.append(tokenized_lnum)
        self.tokenized_lnums = tokenized_lnums
        return tokenized_lnums

    def aggregate_ocr(self):
        max_size = 0
        for i, l_num in enumerate(self.tokenize_lnums()):


            # if the license number is empty reject it
            if len(l_num) == 0 or (len(l_num) == 1 and len(l_num[0]) == 0):
                self.tokenized_lnums.remove(l_num)
                continue

            # if the first symbol is numeric, reject the number
            try:
                first_symbol = int(l_num[0])
                self.tokenized_lnums.remove(l_num)
                continue
            except Exception as e:
                pass

            # calculate max_size
            max_size = max(max_size, len(l_num))


        # initialize the best value aggregation set
        best_values = [list() for i in range(max_size)]

        #perform the aggregation
        for i, l_num in enumerate(self.tokenized_lnums):

            for j, char in enumerate(l_num):
                best_values[j].append(char)


        for char_pos, b_val_list in enumerate(best_values):
            count = dict()
            for char in b_val_list:
                if char in count.keys(): # if already present
                    count[char] += 1 # increment count
                else:
                    count[char] = 1 # otherwise initialize count

            max_val = max(count.values())
            optimum_char = list(count.keys())[list(count.values()).index(max_val)]
            self.license_number = self.license_number + optimum_char

        return self.license_number

