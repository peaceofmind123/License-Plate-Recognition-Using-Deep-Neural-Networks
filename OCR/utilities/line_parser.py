import pandas as pd


def parse_line(bbox_list):
    """
    parse bbox of each characters and align them with their respective lines
    :param bbox_list: list of suspected bboxes
    :return: array of bbox (x1,y1,x2,y2) labelled according line of sentences

    """
    df = pd.DataFrame(bbox_list, columns=['y1', 'x1', 'h', 'w'])
    df = df.sort_values(by=['x1', 'y1'])
    df['y2'] = df['y1'] + df['h']
    df['x2'] = df['x1'] + df['w']
    df = df.drop(columns=['h', 'w'])

    lines = [[tuple(df.iloc[0, [1, 0, 3, 2]].values)]]

    for row_no in range(len(df.values))[1:]:
        x1, y1, x2, y2 = df.iloc[row_no, [1, 0, 3, 2]].values
        added = False

        for line_no in range(len(lines)):
            x1_, y1_, x2_, y2_ = lines[line_no][-1]

            y1_min, y1_max = min(y1, y1_), max(y1, y1_)
            y2_min, y2_max = min(y2, y2_), max(y2, y2_)

            intersection_over_union = (y2_min - y1_max) / (y2_max - y1_min)

            if intersection_over_union > .6:
                lines[line_no].append((x1, y1, x2, y2))
                added = True
                break

        if not added:
            lines.append([(x1, y1, x2, y2)])

    return lines
