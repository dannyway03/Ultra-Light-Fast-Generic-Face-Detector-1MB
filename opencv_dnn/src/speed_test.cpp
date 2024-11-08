// main.cpp
// UltraFaceTest
//
// Created by vealocia on 2019/10/17.
// Copyright © 2019 vealocia. All rights reserved.
//

#include "cv_dnn_ultraface.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int
main(int argc, char **argv)
{
  // Print the OpenCV version
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  // Check if OpenCV is built with CUDA support
  bool isCudaAvailable = cv::cuda::getCudaEnabledDeviceCount() > 0;

  if(isCudaAvailable)
    std::cout << "OpenCV is built with CUDA support." << std::endl;
  else
    std::cout << "OpenCV is NOT built with CUDA support." << std::endl;

  // std::string model_path = argv[1];
  std::string model_path = "../../../models/onnx/version-RFB-640.onnx";
  UltraFace ultraface(model_path, 640, 480, 4, 0.6, 0.2); // Config model input

  // Read from video file
  std::string imgfile
      = "/home/nicola/Software/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
        "imgs/1.jpg";

  cv::namedWindow("UltraFace",
                  cv::WINDOW_NORMAL); // Create a resizable window

  cv::Mat frame = cv::imread(imgfile);

  // Variables to track FPS
  double minFPS = std::numeric_limits<double>::max();
  double maxFPS = 0.0;
  std::vector<double> fpsValues;

  while(true)
    {
      auto start = std::chrono::high_resolution_clock::now();

      try
        {
          std::vector<FaceInfo> face_list;
          ultraface.detect(frame, face_list);

          float maxConf = 0;
          cv::Rect bestFaceRect;
          for(const auto &face : face_list)
            {
              if(face.score > maxConf)
                {
                  maxConf = face.score;
                  bestFaceRect = cv::Rect(face.x1, face.y1, face.x2 - face.x1,
                                          face.y2 - face.y1);
                }
            }
          cv::rectangle(frame, bestFaceRect, cv::Scalar(0, 255, 0), 2);
        }
      catch(const cv::Exception &e)
        {
          std::cerr << "OpenCV error: " << e.what() << std::endl;
        }

      /*for (const auto& face : face_info) {
          cv::Point pt1(face.x1, face.y1);
          cv::Point pt2(face.x2, face.y2);
          cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
      }*/

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      // std::cout << "FPS: " << 1.0 / elapsed.count() << std::endl;
      //  Calculate FPS
      double fps = 1.0 / elapsed.count();
      std::cout << "FPS: " << fps << std::endl;
      std::cout << frame.size << std::endl;

      fpsValues.push_back(fps);
      // Update min and max FPS
      if(fps < minFPS)
        minFPS = fps;
      if(fps > maxFPS)
        maxFPS = fps;

      // Calculate average FPS
      double sumFPS = 0.0;
      for(double fps : fpsValues)
        {
          sumFPS += fps;
        }
      double avgFPS = sumFPS / fpsValues.size();

      // Output FPS statistics
      std::cout << "Min FPS: " << minFPS << std::endl;
      std::cout << "Max FPS: " << maxFPS << std::endl;
      std::cout << "Avg FPS: " << avgFPS << std::endl;

      cv::imshow("UltraFace", frame);
      if(cv::waitKey(1) == 27)
        { // Press 'Esc' to exit
          break;
        }
    }

  cv::destroyAllWindows();

  return 0;
}
