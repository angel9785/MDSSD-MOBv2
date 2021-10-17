from PIL import Image, ImageEnhance
def calculate_mAP_tr(h,bir,det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties,track_boxes,track_success):

    true_images = list()
    # for i in range(len(true_labels)):
    true_images.extend([0] * len(true_labels))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = true_boxes  # (n_objects, 4)
    true_labels = true_labels  # (n_objects)
    true_difficulties = true_difficulties  # (n_objects)
    det_images = list()
    # for i in range(len(det_labels)):
    det_images.extend([0] * len(det_labels))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = det_boxes  # (n_detections, 4)
    det_labels = det_labels  # (n_detections)
    det_scores = det_scores  # (n_detections)

    global n_easy_class_objects

    n_classes = len(label_map)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bir = Image.open(bir, mode='r')
    bir = FT.resize(bir, (512, 640))
    image = np.ascontiguousarray(bir)
    # image=bir
    shapes=np.zeros_like(image,np.uint8)


    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    # det_images = list()
    # for i in range(len(det_labels)):
    #     det_images.extend([i] * det_labels[i].size(0))
    # det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    # det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    # det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    # det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precision = torch.zeros((n_classes - 1), dtype=torch.float)
    recall = torch.zeros((n_classes - 1), dtype=torch.float)
    true_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    false_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    # pre=torch.zeros((n_classes - 1), dtype=torch.float)
    # f = plt.figure()
    for i in range (track_boxes.size(0)):
        track_boxes.append(track_boxes[i].unsqueeze(0))  # (n_class_detections, 4)
        track_success.append(track_success[i].unsqueeze(0))
    if track_boxes.size(0) == 0:
        # continue torch.Tensor()
        track_boxes = torch.Tensor()  # (n_detections, 4)
        track_success = torch.Tensor()
    else:
        track_boxes = torch.cat(track_boxes, dim=0)  # (n_detections, 4)
        track_success = torch.cat(track_success, dim=0)  # (n_detections)


    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images1 = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes1 = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties1 = true_difficulties[true_labels == c]  # (n_class_objects)
        n_class_objects1 = true_class_boxes1.size(0)

        true_class_images3 = list()
        true_class_boxes3 = list()
        true_class_difficulties3 = list()
        true_class_images1 = true_class_images1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)
        scale1 = 0 / 512
        scale = 100 / 512
        for i in range(0, n_class_objects1):

            true_class_images3.append(true_class_images1[i])  # (n_class_detections)
            true_class_boxes3.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
            true_class_difficulties3.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images3 = torch.LongTensor(true_class_images3).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images3.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes3 = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties3 = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes3 = torch.cat(true_class_boxes3, dim=0)  # (n_detections, 4)
            true_class_difficulties3 = torch.cat(true_class_difficulties3, dim=0)  # (n_detections)
        mask=shapes.astype(bool)
        true_class_images5 = list()
        true_class_boxes5 = list()
        true_class_difficulties5 = list()
        # true_class_images1 = true_class_images1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        # true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)
        scale1 = 0/512
        scale = 100 / 512
        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) < scale):
                true_class_images5.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes5.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties5.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images5 = torch.LongTensor(true_class_images5).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images5.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes5 = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties5 = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes5 = torch.cat(true_class_boxes5, dim=0)  # (n_detections, 4)
            true_class_difficulties5 = torch.cat(true_class_difficulties5, dim=0)  # (n_detections)
        # wt=true_class_boxes1[:][1]-true_class_boxes1[:][0]
        # ht=true_class_boxes1[:][3]-true_class_boxes1[:][2]
        true_class_images = list()
        true_class_boxes = list()
        true_class_difficulties = list()
        # true_class_images1 = true_class_images1.unsqueeze(1)
        # # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        # true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)

        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) >= scale):
                true_class_images.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images = torch.LongTensor(true_class_images).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes = torch.cat(true_class_boxes, dim=0)  # (n_detections, 4)
            true_class_difficulties = torch.cat(true_class_difficulties, dim=0)  # (n_detections)

        # (n_class_detections)

        n_easy_class_objects = (true_class_difficulties==0).sum().item()  # ignore difficult objects
        # n_easy_class_objects = (len(true_class_difficulties))
        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)
        true_class_boxes_detected5 = torch.zeros((true_class_difficulties5.size(0)), dtype=torch.uint8).to(
            device)
        # Extract only detections with this class
        det_class_images1 = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes1 = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores1 = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections1 = det_class_boxes1.size(0)
        # wd=det_class_boxes1[:][1]-det_class_boxes1[:][0]
        # hd=det_class_boxes1[:][3]-det_class_boxes1[:][2]

        det_class_images3 = list()
        det_class_boxes3 = list()
        det_class_scores3 = list()
        det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            # if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] >= scale)):
            det_class_images3.append(det_class_images1[i])  # (n_class_detections)
            det_class_boxes3.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
            det_class_scores3.append(det_class_scores1[i].unsqueeze(0))
        det_class_images3 = torch.LongTensor(det_class_images3).to(device)  # (n_detections)
        n_class_detections3 = det_class_images3.size(0)

        det_class_images2 = list()
        det_class_boxes2 = list()
        det_class_scores2 = list()
        # det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] < scale)):
                det_class_images2.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes2.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores2.append(det_class_scores1[i].unsqueeze(0))
        det_class_images2 = torch.LongTensor(det_class_images2).to(device)  # (n_detections)
        n_class_detections2 = det_class_images2.size(0)
        # true_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        # false_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)

        det_class_images = list()
        det_class_boxes = list()
        det_class_scores = list()
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1]>= scale)):
                det_class_images.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores.append(det_class_scores1[i].unsqueeze(0))
        det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
        # n_class_detections = det_class_images.size(0)
        true_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        n_class_detections = det_class_images.size(0)
        # if n_class_detections == 0 & n_class_detections2 == 0:
        #     continue

        if n_class_detections2 !=0:

        # In the order of decreasing scores, check if true or false positive
            det_class_boxes2 = torch.cat(det_class_boxes2, dim=0)  # (n_detections, 4)
            det_class_scores2 = torch.cat(det_class_scores2, dim=0)  # (n_detections)

            # Sort detections in decreasing order of confidence/scores
            det_class_scores2, sort_ind2 = torch.sort(det_class_scores2, dim=0, descending=True)  # (n_class_detections)
            det_class_images2 = det_class_images2[sort_ind2]  # (n_class_detections)
            det_class_boxes2 = det_class_boxes2[sort_ind2]  # (n_class_detections, 4)


        if n_class_detections !=0:
            det_class_boxes = torch.cat(det_class_boxes, dim=0)  # (n_detections, 4)
            det_class_scores = torch.cat(det_class_scores, dim=0)  # (n_detections)

            # Sort detections in decreasing order of confidence/scores
            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
            det_class_images = det_class_images[sort_ind]  # (n_class_detections)
            det_class_boxes = det_class_boxes[sort_ind]

        if n_class_detections3 !=0:
            det_class_boxes3 = torch.cat(det_class_boxes3, dim=0)  # (n_detections, 4)
            det_class_scores3 = torch.cat(det_class_scores3, dim=0)  # (n_detections)

            # Sort detections in decreasing order of confidence/scores
            det_class_scores3, sort_ind3 = torch.sort(det_class_scores3, dim=0, descending=True)  # (n_class_detections)
            det_class_images3 = det_class_images3[sort_ind3]  # (n_class_detections)
            det_class_boxes3 = det_class_boxes3[sort_ind3]
        s1=int(75/512)

        original_dims = torch.FloatTensor(
            [bir.width, bir.height, bir.width, bir.height]).unsqueeze(0)
        # if true_class_boxes.size(0) != 0:
        #     object_boxes_d = true_class_boxes.cpu() * original_dims.cpu()
        #     for i in range(object_boxes_d.size(0)):
        #         box_location = object_boxes_d[i].tolist()
        #         cv2.rectangle(image, (int(box_location[0]), int(box_location[1])),
        #                               (int(box_location[2]), int(box_location[3])), (0, 255, 0), 1)
        # #
        # if true_class_boxes5.size(0) != 0:
        #     object_boxes_d = true_class_boxes5.cpu() * original_dims.cpu()
        #     for i in range(object_boxes_d.size(0)):
        #         box_location = object_boxes_d[i].tolist()
        #         cv2.rectangle(image, (int(box_location[0]), int(box_location[1])),
        #                               (int(box_location[2]), int(box_location[3])), (255, 255, 255),1)

        image1=image.copy()
        for d in range(n_class_detections1):
            if d<n_class_detections:
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar
                det_scores = str(round(det_class_scores[d].to('cpu').tolist(), 2))
                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                # object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

                object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
                # object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    if object_boxes5.size(0) == 0:
                        false_positives[d] = 1
                        this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                        box_location1 = this_detection_box1[0].tolist()
                        cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                      (int(box_location1[2]), int(box_location1[3])),(0, 0, 255),
                                      cv2.FILLED)
                        image=cv2.addWeighted(image1,0.4,image,0.6,0)
                        cv2.putText(image, det_scores,
                                    org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                    fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                    lineType=cv2.LINE_AA)
                        continue

                # Find maximum overlap of this detection with objects in this image of this class
                if object_boxes5.size(0) != 0:
                    overlaps5, overlaps51 = find_jaccard_overlap(this_detection_box, object_boxes5)  # (1, n_class_objects_in_img)
                    max_overlap5, ind5 = torch.max(overlaps5.squeeze(0), dim=0)  # (), () - scalars
                    # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                    # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                    # We need 'original_ind' to update 'true_class_boxes_detected'
                    original_ind5 = torch.LongTensor(range(true_class_boxes5.size(0)))[true_class_images5== this_image][ind5]
                if object_boxes.size(0) != 0:
                    overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
                                                               object_boxes)  # (1, n_class_objects_in_img)
                    max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                    original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                    if ((max_overlap.item() > 0.5) or ((object_boxes[ind][3]-object_boxes[ind][1])<s1 & (max_overlap.item() > 0.1))):
                        # If the object it matched with is 'difficult', ignore it
                        # if object_difficulties[ind] == 0:
                            # If this object has already not been detected, it's a true positive
                        if true_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1

                            true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                        else:
                            false_positives[d] = 1
                            this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                            box_location1 = this_detection_box1[0].tolist()
                            cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                          (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                          cv2.FILLED)
                            image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                            cv2.putText(image, det_scores,
                                        org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                        fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                        lineType=cv2.LINE_AA)
                    # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                    else:
                        if((object_boxes[ind][2]-object_boxes[ind][0])/(object_boxes[ind][3]-object_boxes[ind][1])>=0.7) :
                            if (max_overlap>=0.1):
                                # if object_difficulties[ind] == 0:
                                true_positives[d] = 1

                            else:
                                if object_boxes5.size(0) != 0:
                                    if ((max_overlap5.item() > 0.5) or (
                                            (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (
                                            max_overlap5.item() > 0.1))):

                                        if true_class_boxes_detected5[original_ind5] == 0:
                                            # true_positives[d] = 1
                                            true_class_boxes_detected5[
                                                original_ind5] = 1  # this object has now been detected/accounted for
                                        # Otherwise, it's a false positive (since this object is already accounted for)
                                        else:
                                            false_positives[d] = 1
                                            this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                            box_location1 = this_detection_box1[0].tolist()
                                            cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                          (int(box_location1[2]), int(box_location1[3])),
                                                          (0, 0, 255),
                                                          cv2.FILLED)
                                            image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                            cv2.putText(image, det_scores,
                                                        org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                        fontFace=font, fontScale=1 / 2, color=(0, 0, 255),
                                                        thickness=2,
                                                        lineType=cv2.LINE_AA)
                                    else:
                                        if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                                object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                            # if object_difficulties5[ind5] == 0:
                                            if (max_overlap5 >= 0.1):

                                                print("")
                                            else:
                                                false_positives[d] = 1
                                                this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                                box_location1 = this_detection_box1[0].tolist()
                                                cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                              (int(box_location1[2]), int(box_location1[3])),
                                                              (0, 0, 255),
                                                              cv2.FILLED)
                                                image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                                cv2.putText(image, det_scores,
                                                            org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                            fontFace=font, fontScale=1 / 2, color=(0, 0, 255),
                                                            thickness=2,
                                                            lineType=cv2.LINE_AA)
                                        else:
                                            false_positives[d] = 1
                                            this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                            box_location1 = this_detection_box1[0].tolist()
                                            cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                          (int(box_location1[2]), int(box_location1[3])),
                                                          (0, 0, 255),
                                                          cv2.FILLED)
                                            image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                            cv2.putText(image, det_scores,
                                                        org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                        fontFace=font, fontScale=1 / 2, color=(0, 0, 255),
                                                        thickness=2,
                                                        lineType=cv2.LINE_AA)
                                else:
                                    false_positives[d] = 1
                                    this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                    box_location1 = this_detection_box1[0].tolist()
                                    cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                  (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                                  cv2.FILLED)
                                    image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                    cv2.putText(image, det_scores,
                                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                                lineType=cv2.LINE_AA)
                        else:
                            if object_boxes5.size(0) != 0:
                                if ((max_overlap5.item() > 0.5) or ((object_boxes5[ind5][3]-object_boxes5[ind5][1])<s1 & (max_overlap5.item() > 0.1))):
                                    # If the object it matched with is 'difficult', ignore it
                                    # if object_difficulties5[ind5] == 0:
                                        # If this object has already not been detected, it's a true positive
                                    if true_class_boxes_detected5[original_ind5] == 0:
                                        # true_positives[d] = 1
                                        true_class_boxes_detected5[
                                            original_ind5] = 1  # this object has now been detected/accounted for
                                    # Otherwise, it's a false positive (since this object is already accounted for)
                                    else:
                                        false_positives[d] = 1
                                        this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                        box_location1 = this_detection_box1[0].tolist()
                                        cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                      (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                                      cv2.FILLED)
                                        image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                        cv2.putText(image, det_scores,
                                                    org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                    fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                                    lineType=cv2.LINE_AA)
                                else:
                                    if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (object_boxes5[ind5][3] - object_boxes5[ind5][1])>=0.7):
                                        # if object_difficulties5[ind5] == 0:
                                        if (max_overlap5 >= 0.1):
                                            print("")
                                        else:
                                            false_positives[d] = 1
                                            this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                            box_location1 = this_detection_box1[0].tolist()
                                            cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                          (int(box_location1[2]), int(box_location1[3])),
                                                          (0, 0, 255),
                                                          cv2.FILLED)
                                            image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                            cv2.putText(image, det_scores,
                                                        org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                        fontFace=font, fontScale=1 / 2, color=(0, 0, 255),
                                                        thickness=2,
                                                        lineType=cv2.LINE_AA)
                                    else:
                                        false_positives[d] = 1
                                        this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                        box_location1 = this_detection_box1[0].tolist()
                                        cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                      (int(box_location1[2]), int(box_location1[3])),(0, 0, 255),
                                                      cv2.FILLED)
                                        image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                        cv2.putText(image, det_scores,
                                                    org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                    fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                                    lineType=cv2.LINE_AA)
                            else:
                                false_positives[d] = 1
                                this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                box_location1 = this_detection_box1[0].tolist()
                                cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                              (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                              cv2.FILLED)
                                image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                cv2.putText(image, det_scores,
                                            org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                            fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                            lineType=cv2.LINE_AA)
                else:
                    if object_boxes5.size(0) != 0:
                        if ((max_overlap5.item() > 0.5) or (
                                (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (max_overlap5.item() > 0.1))):
                            # If the object it matched with is 'difficult', ignore it
                            # if object_difficulties5[ind5] == 0:
                            # If this object has already not been detected, it's a true positive
                            if true_class_boxes_detected5[original_ind5] == 0:
                                # true_positives[d] = 1
                                true_class_boxes_detected5[
                                    original_ind5] = 1  # this object has now been detected/accounted for
                            # Otherwise, it's a false positive (since this object is already accounted for)
                            else:
                                false_positives[d] = 1
                                this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                box_location1 = this_detection_box1[0].tolist()
                                cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                              (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                              cv2.FILLED)
                                image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                cv2.putText(image, det_scores,
                                            org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                            fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                            lineType=cv2.LINE_AA)
                        else:
                            if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                    object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                # if object_difficulties5[ind5] == 0:
                                if (max_overlap5 >= 0.1):
                                    print("")
                                else:
                                    false_positives[d] = 1
                                    this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                    box_location1 = this_detection_box1[0].tolist()
                                    cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                                  (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                                  cv2.FILLED)
                                    image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                    cv2.putText(image, det_scores,
                                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                                lineType=cv2.LINE_AA)
                            else:
                                false_positives[d] = 1
                                this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                                box_location1 = this_detection_box1[0].tolist()
                                cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                              (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                              cv2.FILLED)
                                image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                                cv2.putText(image, det_scores,
                                            org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                            fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                            lineType=cv2.LINE_AA)
                    else:
                        false_positives[d] = 1
                        this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
                        box_location1 = this_detection_box1[0].tolist()
                        cv2.rectangle(image1, (int(box_location1[0]), int(box_location1[1])),
                                      (int(box_location1[2]), int(box_location1[3])), (0, 0, 255),
                                      cv2.FILLED)
                        image = cv2.addWeighted(image1, 0.4, image, 0.6, 0)
                        cv2.putText(image, det_scores,
                                                        org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                                        fontFace=font, fontScale=1 / 2, color=(0, 0, 255), thickness=2,
                                                        lineType=cv2.LINE_AA)

        #


        #det=[]
        # for d in range(n_class_detections3):
        #     # if d<n_class_detections:
        #     this_detection_box = det_class_boxes3[d].unsqueeze(0)  # (1, 4)
        #     this_image = det_class_images3[d]  # (), scalar
        #     det_scores = str(round(det_class_scores3[d].to('cpu').tolist(), 2))
        #     # Find objects in the same image with this class, their difficulties, and whether they have been detected before
        #
        #     object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
        #     # object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
        #
        #     object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
        #     # object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
        #     original_dims = torch.FloatTensor(
        #         [bir.width, bir.height, bir.width, bir.height]).unsqueeze(0)
        #     # If no such object in this image, then the detection is a false positive
        #     if object_boxes.size(0) == 0:
        #         if object_boxes5.size(0) == 0:
        #             false_positives[d] = 1
        #             continue
        #
        #     # Find maximum overlap of this detection with objects in this image of this class
        #     if object_boxes5.size(0) != 0:
        #         overlaps5, overlaps51 = find_jaccard_overlap(this_detection_box, object_boxes5)  # (1, n_class_objects_in_img)
        #         max_overlap5, ind5 = torch.max(overlaps5.squeeze(0), dim=0)  # (), () - scalars
        #
        #         original_ind5 = torch.LongTensor(range(true_class_boxes5.size(0)))[true_class_images5== this_image][ind5]
        #     if object_boxes.size(0) != 0:
        #         overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
        #                                                    object_boxes)  # (1, n_class_objects_in_img)
        #         max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
        #         original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
        #         this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
        #         if ((max_overlap.item() > 0.5) or ((object_boxes[ind][3]-object_boxes[ind][1])<s1 & (max_overlap.item() > 0.1))):
        #
        #             if true_class_boxes_detected[original_ind] == 0:
        #                 true_positives[d] = 1
        #                 box_location1 = this_detection_box1[0].tolist()
        #                 cv2.rectangle(image, (int(box_location1[0]), int(box_location1[1])),
        #                                       (int(box_location1[2]), int(box_location1[3])), (255, 255, 0),
        #                                       3)
        #                 det.extend(box_location1)
        #                 cv2.putText(image, det_scores,
        #                                     org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
        #                                     fontFace=font, fontScale=1 / 2, color=(255, 255, 0), thickness=2,
        #                                     lineType=cv2.LINE_AA)
        #                 true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
        #
        #         else:
        #             if((object_boxes[ind][2]-object_boxes[ind][0])/(object_boxes[ind][3]-object_boxes[ind][1])>=0.7):
        #
        #                 if (max_overlap>=0.1):
        #                     true_positives[d] = 1
        #                     box_location1 = this_detection_box1[0].tolist()
        #                     cv2.rectangle(image, (int(box_location1[0]), int(box_location1[1])),
        #                                           (int(box_location1[2]), int(box_location1[3])), (255, 255, 0),
        #                                           3)
        #                     det.extend(box_location1)
        #                     cv2.putText(image, det_scores,
        #                                         org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
        #                                         fontFace=font, fontScale=1 / 2, color=(255, 255, 0), thickness=2,
        #                                         lineType=cv2.LINE_AA)
        #
        # # #
        m=[]
        for d in range(det.size(0)):
            m.append(det[d].unsqueeze(0))
        m=torch.cat(m,dim=0)

        for d in range(track_success.shape):
            # if d<n_class_detections:
            this_detection_box = track_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images3[d]  # (), scalar
            track_success = str(round(track_success[d].to('cpu').tolist(), 2))
            # Find objects in the same image with this class, their difficulties, and whether they have been detected before

            # object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            # object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

            # object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
            # object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
            original_dims = torch.FloatTensor(
                [bir.width, bir.height, bir.width, bir.height]).unsqueeze(0)
            # If no such object in this image, then the detection is a false positive
            # if object_boxes.size(0) == 0:
            #     if object_boxes5.size(0) == 0:
            #         false_positives[d] = 1
            #         continue

            # Find maximum overlap of this detection with objects in this image of this class



            if m.size(0) != 0:
                overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
                                                           m)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                this_detection_box1 = this_detection_box.cpu()
                if ((max_overlap.item() > 0.5) ):

                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        box_location1 = this_detection_box1[0].tolist()
                        cv2.rectangle(image, (int(box_location1[0]), int(box_location1[1])),
                                              (int(box_location1[2]), int(box_location1[3])), (255, 255, 255),
                                              3)
                        cv2.putText(image, track_success,
                                            org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                            fontFace=font, fontScale=1 / 2, color=(255, 255, 255), thickness=1,
                                            lineType=cv2.LINE_AA)
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for


    # image = np.ascontiguousarray(image)
    cv2.imwrite('/home/fereshteh/result55/' + str(h) + '.png', image)


    return c