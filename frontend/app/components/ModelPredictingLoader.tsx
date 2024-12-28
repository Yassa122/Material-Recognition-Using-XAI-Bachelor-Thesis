// components/ModelTrainingLoader.tsx
"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import StarIcon from "@/public/Star.svg"; // Adjust the path as needed

const ModelTrainingLoader = () => {
  const [progress, setProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Starting training..."
  );
  const [estimatedTimeLeft, setEstimatedTimeLeft] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const router = useRouter();

  useEffect(() => {
    let interval: NodeJS.Timeout;

    const fetchTrainingStatus = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/get_prediction_status"
        );
        const data = await response.json();

        setStatusMessage(data.message || "Predicting in progress...");

        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.eta !== undefined) {
          setEstimatedTimeLeft(data.eta);
        }

        // Redirect when progress reaches 100%
        if (data.progress === 100) {
          setLoading(false);
          clearInterval(interval);
          router.push("/pages/predictions"); // Redirect to /predictions
        }

        // Optionally, handle the status "completed" if it's separate from progress
        if (data.status === "completed") {
          setLoading(false);
          clearInterval(interval);
          router.push("/pages/predictions"); // Redirect to /predictions
        } else if (data.status === "error") {
          setLoading(false);
          clearInterval(interval);
          // Handle error state appropriately
          console.error("Prediction error:", data.message);
          // Optionally, display an error message to the user
        }
      } catch (error) {
        console.error("Error fetching predicting status:", error);
        // Optionally, handle fetch errors (e.g., network issues)
      }
    };

    // Initial fetch
    fetchTrainingStatus();

    // Set up interval to fetch status every 5 seconds
    interval = setInterval(fetchTrainingStatus, 5000); // Fetch every 5 seconds

    // Clean up the interval on component unmount
    return () => clearInterval(interval);
  }, [router]);

  return (
    <div className="bg-[#202020] p-16 rounded-lg shadow-lg flex flex-col items-center justify-center space-y-6 w-full h-full">
      {/* Animated SVG Icon */}
      <div className="bg-[#181818] p-6 rounded-full animate-pulse">
        <Image
          src={StarIcon}
          alt="Loading Icon"
          width={84}
          height={84}
          className="animate-spin-slow"
        />
      </div>

      {/* Title */}
      <h2 className="text-3xl font-bold text-white">Model Predicting...</h2>
      <p className="text-gray-400 text-lg">{statusMessage}</p>

      {/* Progress Bar */}
      <div className="w-full mt-8">
        <div className="h-4 rounded-full bg-gray-700 overflow-hidden relative">
          <div
            className="h-4 rounded-full bg-gradient-to-r from-green-400 via-yellow-500 to-red-600"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Progress Percentage and Estimated Time Left */}
      <div className="mt-4 text-gray-300">
        <p>Progress: {Math.round(progress)}%</p>
        {estimatedTimeLeft && <p>Estimated Time Left: {estimatedTimeLeft}</p>}
      </div>
    </div>
  );
};

export default ModelTrainingLoader;
