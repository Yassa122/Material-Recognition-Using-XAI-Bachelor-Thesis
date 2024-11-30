"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import StarIcon from "@/public/Star.svg"; // Adjust the path as needed

const ModelTrainingLoader = () => {
  const [progress, setProgress] = useState<number>(0);
  const [estimated_time_left, setTimeLeft] = useState<string>("2h 15m");
  const [loading, setLoading] = useState<boolean>(true);
  const router = useRouter();

  // Constants for training time (2 hours 15 minutes)
  const totalTrainingTime =   60 + 15; // Total training time in minutes (135 minutes)

  useEffect(() => {
    let interval: NodeJS.Timeout;
    let startTime = Date.now(); // Track the time when the simulation starts
    const simulateProgress = () => {
      interval = setInterval(() => {
        // Simulate progress increase
        const elapsedTimeInMinutes = (Date.now() - startTime) / 1000 / 60; // in minutes
        const currentProgress = Math.min(
          (elapsedTimeInMinutes / totalTrainingTime) * 100,
          100
        );

        setProgress(currentProgress);

        // Calculate estimated time left based on current progress
        const timeLeftInMinutes = totalTrainingTime - elapsedTimeInMinutes;
        const minutesLeft = Math.floor(timeLeftInMinutes);
        const secondsLeft = Math.floor((timeLeftInMinutes - minutesLeft) * 60);
        setTimeLeft(`${minutesLeft}m ${secondsLeft}s`);

        // If the training reaches 100%, stop the simulation and redirect
        if (currentProgress >= 100) {
          clearInterval(interval); // Stop the simulation
          setLoading(false); // Hide the loading spinner
          router.push("/pages/model-training"); // Redirect to model training page
        }
      }, 5000); // Update every 5 seconds
    };

    simulateProgress();

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
      <h2 className="text-3xl font-bold text-white">Model Training...</h2>
      <p className="text-gray-400 text-lg">Your data is being loaded...</p>

      {/* Glowing and Animated Loading Bar */}
      <div className="w-full mt-8">
        <div className="h-4 rounded-full bg-gray-700 overflow-hidden relative">
          <div
            className="h-4 rounded-full bg-gradient-to-r from-green-400 via-yellow-500 to-red-600"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Progress and Time Left */}
      <div className="mt-4 text-gray-300">
        <p>Progress: {Math.round(progress)}%</p>
        <p>Time Left: {estimated_time_left}</p>
      </div>
    </div>
  );
};

export default ModelTrainingLoader;
