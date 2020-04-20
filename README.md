# Cat Sprayer

## Introduction

Recently I adopted a cat from a rescue center, who arrived with some bad habits. 
The worst of these was jumping onto the kitchen counter, risking getting hurt by the hot stove, 
or eating something dangerous for cats. In this problem, I saw an opportunity for a pet project. 
My solution was to build an edge device that uses computer vision to detect when the cat is 
on the counter and give her a light spray of water. In addition to spraying the cat, the device will 
also send a message to my phone with a video clip of the event and log the event to an SQL database. 
The content of this database is visualized through a React frontend, highlighting changes in cat’s behavior 
over time (we’ve seen significant improvements to date). The ML detection, SQL database, Flask API and React 
application are all hosted on a single raspberry pi.

This repository contains all the code required to run the device and send video clips to your mobile device, 
although this will require that you also setup an aws and Twilio account. 
The Twilio API is used to send the message while s3 is used to store clips and 
generate the pre-signed urls required for the Twilio api. 

Below is a sample gif, created from one of the video events that was sent to my phone! 

![Sample Gif](images/sample_video.gif)

## Set-up

The project is highly configurable, everything can be controlled from config.py.

To run this code you should set up a virtual environment on your raspberry pi, then ensure that all environmental 
variables mentioned in env_setup are set and install all the requirements listed in requirements.txt. 

## The hardware 

A full list of hardware and instructions on how to build your own is coming soon, 
this will include a 3D model of the new enclosure which you can print for yourself! 


## The build 

Below is an image of the very first proto-type of the device. Here a simple food container 
is being used as the devices housing. The next iteration will use a custom 3D-printer housing with 
an integral water reservoir.

![ProtoType Housing](images/device_proto_1.jpeg)


The water pump used in this project is a simple 3.7V pump/motor combo salvaged from an electric plant mister, 
I was also able to salvage the battery from this to act as a UPS and to provide the current required 
by the motor. When plugged into the wall the batter is trickle charged via a very cheap charge controller.  

![Water Pump](images/motor_salvage.jpeg)


It was very important to test the current draw of the pump under full load so I could correctly spec out 
the hardware controller circuit. I could the pump was drawing a max of 1.5 amps when there was a blockage at 
the water inlet of the pipe. In my prototype I'm controlling the motor with a Tip 120, this is controlled by pin 18
of the raspberry. It's important to include a 'fly back' diode across the motor to protect the pi and a 1k resistor 
on the pin controlling the Tip120. It's also a good idea to use a 10k tie-down resistor to ensure the pump is not 
triggered during startup/shutdown of the pi.   

![Current Testing](images/current_testing.jpeg)
