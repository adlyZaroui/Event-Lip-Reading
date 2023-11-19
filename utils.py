''' Description: Utility functions for the project. '''



def load_data(base_path, class_folders=None):
    
    '''
    Loads the data from the specified base path.
    
    Parameters:
        base_path (str): The base path to the data.
        class_folders (list): The list of class folders names to load, or None to load all.
        
    Returns:
        dataframes (list): A list of dataframes.
    '''
    
    from tqdm import tqdm # for tracking progress
    import os
    import pandas as pd
    
    # Initialize a list to hold dataframes
    dataframes = []

    if class_folders:
        # Loop over each class folder
        for folder in tqdm(class_folders):
            # Get the list of CSV files in this class folder
            csv_files = os.listdir(os.path.join(base_path, folder))
            
            # Loop over each CSV file
            for csv_file in csv_files:
                # Define the full path to the CSV file
                csv_path = os.path.join(base_path, folder, csv_file)
                
                # Load the CSV file into a dataframe
                df = pd.read_csv(csv_path)
                
                # Append the dataframe to the list
                dataframes.append(df)
    else:
        # Get the list of CSV files in the base path
        csv_files = os.listdir(base_path)

        # Loop over each CSV file
        for csv_file in tqdm(csv_files):
            # Define the full path to the CSV file
            csv_path = os.path.join(base_path, csv_file)
            
            # Load the CSV file into a dataframe
            df = pd.read_csv(csv_path)
            
            # Append the dataframe to the list
            dataframes.append(df)

    return dataframes


def events_to_image(df, x_max=640, y_max=480):
    
    '''
    
    Converts a dataframe of events to a 2D image taht is a 2D histogram of the events.
    
    Args:
        df (pd.DataFrame): A dataframe of events.
        x_max (int): The maximum x coordinate, defaults to 100.
        y_max (int): The maximum y coordinate, defaults to 100.
        
    Returns:
        np.ndarray: A 2D image.
    
    '''
    
    from scipy.ndimage import rotate
    import numpy as np
    
    # Create a 2D histogram of the event data
    hist, _, _ = np.histogram2d(df['x'], df['y'], bins=[x_max, y_max], weights=df['polarity'])

    # Normalize the histogram to the range [0, 255]
    hist = 255 * (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
    
    hist = rotate(hist, 225)

    return hist.astype(np.uint8)

def event_agg_no_polarity(x, y, polarity, timestamp, T_r=100000, M=640, N=480): # Doesn't take into account polarity
    '''
    Aggregate events into superframes.
    
    Args:
        x (np.array): x coordinates of events
        y (np.array): y coordinates of events
        polarity (np.array): polarity of events
        timestamp (np.array): timestamp of events
        T_r (float): time interval of superframes, defaults to 100000.
        M (int): image length, defaults to 640.
        N (int): image width, defaults to 480.
        
    Returns:
        superframes (np.array): superframes
    '''
    
    from scipy.ndimage import rotate
    import numpy as np
    from tqdm import tqdm
    
    T_seq = timestamp.max()
    T_frames = int((T_seq // T_r)) + 1
    
    frames_0 = np.zeros((T_frames, M, N)) # polarity == 0
    frames_1 = np.zeros((T_frames, M, N)) # polarity == 1
    
    for i in tqdm(range(T_frames)):
        idx_0 = np.where((timestamp >= i * T_r) & (timestamp < (i+1) * T_r) & (polarity == 0))[0]
        if len(idx_0) > 0:
            frames_0[i] = np.bincount(N * x[idx_0] + y[idx_0], minlength = M * N).reshape(M, N)
        
        idx_1 = np.where((timestamp >= i * T_r) & (timestamp < (i+1) * T_r) & (polarity == 1))[0]
        if len(idx_1) > 0:
            frames_1[i] = np.bincount(N * x[idx_1] + y[idx_1], minlength = M * N).reshape(M, N)
    
    # Rotate the frames
    frames_0 = rotate(frames_0, 225, axes=(1,2), reshape=False)
    frames_1 = rotate(frames_1, 225, axes=(1,2), reshape=False)
    
    superframes = np.concatenate((frames_0, frames_1), axis = 0)
    print('generated superframes with size:', superframes.shape)
    return superframes



def event_agg_polarity(x, y, polarity, timestamp, T_r=100000, M=640, N=480): # Takes into account polarity
    '''
    Aggregate events into superframes.
    
    Args:
        x (np.array): x coordinates of events
        y (np.array): y coordinates of events
        polarity (np.array): polarity of events
        timestamp (np.array): timestamp of events
        T_r (float): time interval of superframes, defaults to 100000.
        M (int): image length, defaults to 640.
        N (int): image width, defaults to 480.
        
    Returns:
        superframes_0 (np.array): superframes for polarity 0
        superframes_1 (np.array): superframes for polarity 1
    '''
    
    from scipy.ndimage import rotate
    import numpy as np
    from tqdm import tqdm
    
    T_seq = timestamp.max()
    T_frames = int((T_seq // T_r)) + 1
    
    frames_0 = np.zeros((T_frames, M, N)) # polarity == 0
    frames_1 = np.zeros((T_frames, M, N)) # polarity == 1
    
    for i in tqdm(range(T_frames)):
        idx_0 = np.where((timestamp >= i * T_r) & (timestamp < (i+1) * T_r) & (polarity == 0))[0]
        if len(idx_0) > 0:
            frames_0[i] = np.bincount(N * x[idx_0] + y[idx_0], minlength = M * N).reshape(M, N)
        
        idx_1 = np.where((timestamp >= i * T_r) & (timestamp < (i+1) * T_r) & (polarity == 1))[0]
        if len(idx_1) > 0:
            frames_1[i] = np.bincount(N * x[idx_1] + y[idx_1], minlength = M * N).reshape(M, N)
    
    # Rotate the frames
    frames_0 = rotate(frames_0, 225, axes=(1,2), reshape=False)
    frames_1 = rotate(frames_1, 225, axes=(1,2), reshape=False)
    
    print('generated superframes with size:', frames_0.shape, 'and', frames_1.shape)
    return frames_0, frames_1

    
def decompose_events(event_df):
    '''
    
    Decompose the events dataframe into tuple of individual columns x, y, polarity, and timestamp.
    
    Args:
        test_data (pd.DataFrame): A dataframe of events.
        
    Returns:
        4-tuple: x (np.array): x coordinates of events,
                 y (np.array): y coordinates of events,
                 polarity (np.array): polarity of events,
                 timestamp (np.array): timestamp of events  
    '''
    
    import numpy as np
    
    return (np.array(event_df['x'].values),
            np.array(event_df['y'].values),
            np.array(event_df['polarity'].values),
            np.array(event_df['time'].values))