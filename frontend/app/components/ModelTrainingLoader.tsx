"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import StarIcon from "@/public/Star.svg"; // Adjust the path as needed

interface TrainingProgress {
  progress: number; // 0 to 100
  timeLeft: string; // e.g., "2m 30s"
}

const ModelTrainingLoader = () => {
  const [progress, setProgress] = useState<number>(0);
  const [timeLeft, setTimeLeft] = useState<string>("Loading...");
  const [loading, setLoading] = useState<boolean>(true);
  const router = useRouter();

  useEffect(() => {
    // Set up a periodic request to check the training status
    const interval = setInterval(() => {
      // Fetch the current training status from Flask backend
      fetch("http://localhost:5000/get_training_status")
        .then((response) => response.json())
        .then((data: TrainingProgress) => {
          setProgress(data.progress);
          setTimeLeft(data.timeLeft);

          // If the training is complete, stop polling and redirect
          if (data.progress === 100) {
            clearInterval(interval); // Stop polling
            setLoading(false); // Hide the loading spinner
            router.push("/pages/model-training"); // Redirect to model training page
          }
        })
        .catch((error) => {
          console.error("Error fetching training status:", error);
          clearInterval(interval); // Stop polling in case of an error
        });
    }, 5000); // Check the status every 5 seconds

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
        <p>Progress: {progress}%</p>
        <p>Time Left: {timeLeft}</p>
      </div>
    </div>
  );
};

export default ModelTrainingLoader;
