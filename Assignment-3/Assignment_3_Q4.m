% Read the video file
videoFile = 'cv-ass-3_video.mp4';
videoReader = VideoReader(videoFile);

% Parameters for reference frames
referenceFrames = [1, 11, 31];

% Optical flow parameters
opticFlow = opticalFlowFarneback('NumPyramidLevels',3, 'PyramidScale',0.5, 'NumIterations',15, 'NeighborhoodSize',7, 'FilterSize',5);

% Define the output video
outputVideo = VideoWriter('optical_flow.mp4', 'MPEG-4');
outputVideo.FrameRate = videoReader.FrameRate;
open(outputVideo);

% Read the first frame
prevFrame = readFrame(videoReader);
prevGray = rgb2gray(prevFrame);

% Process each frame
while hasFrame(videoReader)
    frame = readFrame(videoReader);
    gray = rgb2gray(frame);
    
    % Loop over the reference frames
    for i = 1:length(referenceFrames)
        if referenceFrames(i) == 1 || mod(videoReader.CurrentTime*videoReader.FrameRate, referenceFrames(i)) == 0
            % Calculate optical flow
            flow = estimateFlow(opticFlow, prevGray);
            
            % Plot optical flow vectors
            imshow(frame);
            hold on;
            plot(flow, 'DecimationFactor', [10 10], 'ScaleFactor', 2);
            hold off;
            
            % Convert figure to frame
            drawnow;
            frameWithFlow = getframe;
            
            % Resize frame to match original frame size
            frameWithFlow = imresize(frameWithFlow.cdata, [size(frame, 1), size(frame, 2)]);
            
            % Write frame with optical flow to video
            writeVideo(outputVideo, frameWithFlow);
        end
    end
    
    % Update previous frame
    prevGray = gray;
end

% Close the video writer
close(outputVideo);
