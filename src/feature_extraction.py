def extract_all_windows(stepSize, windowSize, apply_preprocessing=True):
    '''
    Return all windows for given image
    '''
    image_files = glob.glob(os.path.join(config.DATASET_PATH, '*.JPG'))

    for f in image_files:

        filename = os.path.splitext(os.path.split(f)[1])[0]

        if not os.path.exists(os.path.join(config.WINDOW_PATH, filename)):
            os.makedirs(os.path.join(config.WINDOW_PATH, filename))

        img = cv2.imread(f)
        
        for y in range(0, img.shape[0], stepSize):
            for x in range(0, img.shape[1], stepSize):
            # yield the current window
                window = (x, y, img[y:y + windowSize[1], x:x + windowSize[0]])
                cv2.imwrite('{}/{}.png'.format(os.path.join(config.WINDOW_PATH, filename), uuid4()), window[2])
