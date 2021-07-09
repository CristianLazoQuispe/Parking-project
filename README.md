# Parking-project


# Install python3.7

        $ sudo apt-get update
        $ sudo apt install software-properties-common
        $ sudo add-apt-repository ppa:deadsnakes/ppa
        $ sudo apt install python3.7
        $ python3.7 --version

# Install virtual enviroment

        $ sudo apt install python3.7-venv
        $ python3.7 -m venv env37park
        $ source env37park/bin/activate

# Install basic libraries

        $ pip install matplotlib==3.4.2
        $ pip install matplotlib-inline==0.1.2
        $ pip install PyQt5==5.9.2
        $ pip install Shapely==1.7.1
        $ pip install opencv-python==4.5.2.52

# Install maskrcnn

    https://www.immersivelimit.com/tutorials/mask-rcnn-for-windows-10-tensorflow-2-cuda-101

        $ git clone https://github.com/akTwelve/Mask_RCNN.git aktwelve_mask_rcnn
        $ pip install -r requirements.txt
        $ python setup.py clean --all install


# Download model

        $ cd Codes
        $ python3 download_model.py

# Regions

        $ cd Codes
        $ python3 set_regions.py -v stace_park3.mp4 -o regions.p

        Mark spaces                 
                
                - Click 4 points
                - Push n
                - Push q
        
        Fish process
                
                - Push b

# Detector-img

        $ cd Codes
        $ python3 detector-img.py "../Data/espacio-libre.jpg" "../Results/regions.p"

# Detector-video

        $ cd Codes
        $ python3 detector-video.py "../Data/stace_park3.mp4" "../Results/regions.p"
