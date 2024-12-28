"use client";
import { useState, useEffect, useRef } from "react";
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
  const [showCompletionModal, setShowCompletionModal] =
    useState<boolean>(false); // New state for modal
  const router = useRouter();
  const modalRef = useRef<HTMLDivElement>(null); // Reference for modal

  useEffect(() => {
    let interval: NodeJS.Timeout;

    const fetchTrainingStatus = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/get_training_status"
        );
        const data = await response.json();

        setStatusMessage(data.message || "Training in progress...");

        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.eta !== undefined) {
          setEstimatedTimeLeft(data.eta);
        }

        if (data.status === "completed" || data.progress === 100) {
          // Check for completion
          setLoading(false);
          clearInterval(interval);
          setShowCompletionModal(true); // Show modal instead of redirecting
        } else if (data.status === "error") {
          setLoading(false);
          clearInterval(interval);
          // Handle error state appropriately
          console.error("Training error:", data.message);
          // Optionally, display an error message to the user
        }
      } catch (error) {
        console.error("Error fetching training status:", error);
        // Optionally, handle fetch errors (e.g., network issues)
      }
    };

    // Initial fetch immediately
    fetchTrainingStatus();
    // Set interval to fetch every 5 seconds
    interval = setInterval(fetchTrainingStatus, 5000);

    // Clean up the interval on component unmount
    return () => clearInterval(interval);
  }, [router]);

  // Handler for button click in the modal
  const handleStartPredictions = () => {
    setShowCompletionModal(false); // Close the modal
    // Redirect to data upload page with query parameter
    router.push("/pages/data-upload?autoToggle=true");
  };

  // Handler to close the modal
  const handleCloseModal = () => {
    setShowCompletionModal(false);
  };

  // Handle key presses for accessibility (e.g., Esc key to close modal)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && showCompletionModal) {
        setShowCompletionModal(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showCompletionModal]);

  // Focus trap within the modal
  useEffect(() => {
    if (showCompletionModal && modalRef.current) {
      const focusableElements = modalRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      const handleTabKey = (e: KeyboardEvent) => {
        if (e.key !== "Tab") return;

        if (e.shiftKey) {
          // Shift + Tab
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          // Tab
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      };

      document.addEventListener("keydown", handleTabKey);
      firstElement.focus();

      return () => {
        document.removeEventListener("keydown", handleTabKey);
      };
    }
  }, [showCompletionModal]);

  return (
    <div className="bg-[#202020] p-16 rounded-lg shadow-lg flex flex-col items-center justify-center space-y-6 w-full h-full relative">
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
      <p className="text-gray-400 text-lg">{statusMessage}</p>

      {/* Progress Bar */}
      <div className="w-full mt-8">
        <div className="h-4 rounded-full bg-gray-700 overflow-hidden relative">
          <div
            className="h-4 rounded-full bg-gradient-to-r from-green-400 via-yellow-500 to-red-600 transition-width duration-500"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Progress Percentage and Estimated Time Left */}
      <div className="mt-4 text-gray-300">
        <p>Progress: {Math.round(progress)}%</p>
        {estimatedTimeLeft && <p>Estimated Time Left: {estimatedTimeLeft}</p>}
      </div>

      {/* Completion Modal */}
      {showCompletionModal && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50 transition-opacity duration-300"
          aria-modal="true"
          role="dialog"
          aria-labelledby="modal-title"
          aria-describedby="modal-description"
          onClick={handleCloseModal} // Close when clicking outside the modal content
        >
          <div
            className="bg-[#2D2D2D] p-8 rounded-lg shadow-xl max-w-md w-full relative transform transition-all duration-300"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the modal
            ref={modalRef}
          >
            {/* Close Button */}
            <button
              onClick={handleCloseModal}
              className="absolute top-4 right-4 text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
              aria-label="Close Modal"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>

            <h3
              id="modal-title"
              className="text-2xl font-semibold text-white mb-4"
            >
              🎉 Training Completed!
            </h3>
            <p id="modal-description" className="text-gray-300 mb-6">
              Your model has been successfully trained. You can now start making
              predictions.
            </p>
            <div className="flex justify-center space-x-4">
              <button
                onClick={handleStartPredictions}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Start Predictions
              </button>
              <button
                onClick={handleCloseModal}
                className="px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Maybe Later
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelTrainingLoader;
