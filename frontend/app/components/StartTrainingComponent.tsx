// components/StartTrainingComponent.tsx
"use client";

import React from "react";
import { motion } from "framer-motion";
import { FaRobot, FaPlay } from "react-icons/fa";
import { useRouter } from "next/navigation"; // Import useRouter

const StartTrainingComponent = () => {
  // Removed onStart prop
  const router = useRouter(); // Initialize router

  const handleStart = () => {
    router.push("/pages/data-upload"); // Redirect to /data-upload
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-full bg-neutral-800 p-12 rounded-lg shadow-2xl">
      {/* Icon */}
      <motion.div
        className="mb-8 text-blue-500 dark:text-blue-400"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 100, damping: 10 }}
      >
        <FaRobot className="w-20 h-20" />
      </motion.div>

      {/* Heading */}
      <motion.h2
        className="text-4xl font-extrabold mb-6 text-gray-300 dark:text-gray-200 text-center"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Model Not Trained
      </motion.h2>

      {/* Description */}
      <motion.p
        className="text-xl mb-12 text-gray-400 dark:text-gray-300 text-center max-w-xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        Your AI model is not trained yet. Start the training process to leverage
        advanced AI capabilities and unlock powerful insights.
      </motion.p>

      {/* Start Training Button */}
      <motion.button
        id="start-action-button" // Added ID for the tour
        onClick={handleStart}
        className="flex items-center px-8 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-300 dark:focus:ring-indigo-800 start-training-button"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FaPlay className="mr-3 w-5 h-5" />
        Start Training
      </motion.button>
    </div>
  );
};

export default StartTrainingComponent;
