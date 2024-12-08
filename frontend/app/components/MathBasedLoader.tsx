// components/CalculationLoader.tsx
"use client";

import { useState, useEffect, useRef } from "react";
import { FaExclamationTriangle, FaSyncAlt, FaDownload } from "react-icons/fa";

interface CalculationLoaderProps {
  taskId: string;
  statusEndpoint: string;
  onComplete: (downloadUrl: string) => void;
}

const CalculationLoader: React.FC<CalculationLoaderProps> = ({
  taskId,
  statusEndpoint,
  onComplete,
}) => {
  const [progress, setProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Initializing calculation..."
  );
  const [estimatedTimeLeft, setEstimatedTimeLeft] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState<number>(0);
  const isMounted = useRef<boolean>(true);

  useEffect(() => {
    isMounted.current = true;

    const fetchCalculationStatus = async () => {
      try {
        const response = await fetch(`${statusEndpoint}/${taskId}`);
        if (!response.ok) {
          throw new Error(`Server responded with status ${response.status}`);
        }
        const data = await response.json();

        if (!isMounted.current) return;

        setStatusMessage(data.message || "Calculation in progress...");
        setProgress(data.progress !== undefined ? data.progress : progress);
        setEstimatedTimeLeft(data.eta || "");

        if (data.status === "completed") {
          setDownloadUrl(data.download_url || null);
          onComplete(data.download_url);
        } else if (data.status === "error") {
          throw new Error(
            data.error_detail || "An error occurred during calculation."
          );
        }
      } catch (err: any) {
        if (!isMounted.current) return;

        setError(err.message || "An unknown error occurred.");
      }
    };

    let pollingInterval = 5000; // Start with 5 seconds
    const maxInterval = 60000; // Maximum interval of 60 seconds
    let timer: NodeJS.Timeout;

    const startPolling = () => {
      fetchCalculationStatus();

      timer = setInterval(() => {
        fetchCalculationStatus();
        // Implement exponential backoff
        pollingInterval = Math.min(pollingInterval * 2, maxInterval);
        clearInterval(timer);
        timer = setInterval(fetchCalculationStatus, pollingInterval);
      }, pollingInterval);
    };

    startPolling();

    return () => {
      isMounted.current = false;
      clearInterval(timer);
    };
  }, [statusEndpoint, taskId, onComplete, progress, retryCount]);

  const handleRetry = () => {
    setError(null);
    setRetryCount((prev) => prev + 1);
    setProgress(0);
    setStatusMessage("Retrying calculation...");
  };

  return (
    <div className="bg-gray-900 p-8 rounded-xl shadow-2xl flex flex-col items-center justify-center space-y-6 w-full max-w-md mx-auto">
      {/* Animated Loader */}
      {!downloadUrl && !error && (
        <div className="relative w-24 h-24">
          <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xl font-semibold text-blue-500">🔄</span>
          </div>
        </div>
      )}

      {/* Status Message */}
      <h2 className="text-2xl font-medium text-gray-100 text-center">
        {statusMessage}
      </h2>

      {/* Progress Bar */}
      {!downloadUrl && !error && (
        <div className="w-full mt-4">
          <div className="h-4 rounded-full bg-gray-700 overflow-hidden">
            <div
              className="h-4 bg-blue-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
              role="progressbar"
            ></div>
          </div>
          <div className="flex justify-between mt-1 text-sm text-gray-400">
            <span>{Math.round(progress)}%</span>
            {estimatedTimeLeft && <span>ETA: {estimatedTimeLeft}</span>}
          </div>
        </div>
      )}

      {/* Download Button */}
      {downloadUrl && (
        <div className="flex flex-col items-center space-y-4">
          <p className="text-lg text-green-400">Calculation Completed!</p>
          <a
            href={downloadUrl}
            download
            className="flex items-center bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-md shadow-lg transition duration-300"
          >
            <FaDownload className="mr-2" />
            Download PDF
          </a>
        </div>
      )}

     
    </div>
  );
};

export default CalculationLoader;
