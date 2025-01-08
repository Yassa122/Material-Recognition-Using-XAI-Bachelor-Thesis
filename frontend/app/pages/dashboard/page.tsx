// pages/dashboard.tsx

"use client";
import { useState, useEffect } from "react";
import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header";
import LineChartComponent from "@/app/components/LineChartComponent";
import UploadComponent from "@/app/components/UploadComponent";
import MultiBarLineChartComponent from "@/app/components/MultiBarLineChartComponent";
import ShapExplanationChart from "@/app/components/ShapExplanationChart";
import StackedBarChart from "@/app/components/StackedBarChart";
import HorizontalBarChart from "@/app/components/HorizontalBarChart";
import LimeCharts from "@/app/components/LimeCharts";
import TrainingStatusTable from "@/app/components/TrainingStatusTable";
import StartTrainingComponent from "@/app/components/StartTrainingComponent";
import { motion } from "framer-motion";
import tutorialSteps from "@/app/tutorialSteps"; // Import the steps
import CustomTour from "@/app/components/CustomTour"; // Import the custom tour
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import ModelMetricsCard from "@/app/components/ModelMetricsCard";
import IntegratedGradientsView from "@/app/components/IntegratedGradientsView";
import ActualVsPredictedDynamic from "@/app/components/ActualVsPredictedDynamic";

const DashboardPage = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [isTrained, setIsTrained] = useState<boolean | null>(null); // null indicates loading
  const [isTourOpen, setIsTourOpen] = useState(false); // State to control CustomTour

  useEffect(() => {
    console.log("DashboardPage mounted. Fetching training status...");
    // Fetch training status from your API
    const fetchTrainingStatus = async () => {
      try {
        const response = await fetch(
          "http://127.0.0.1:5000/check-training-status"
        );
        const data = await response.json();
        console.log("Received training status:", data);
        setIsTrained(data.isTrained);
      } catch (error) {
        console.error("Error fetching training status:", error);
        setIsTrained(false); // Default to false on error
      }
    };

    fetchTrainingStatus();

    // Start the tutorial if not completed
    const hasCompletedTutorial = localStorage.getItem("hasCompletedTutorial");
    if (!hasCompletedTutorial) {
      console.log("Tutorial not completed. Opening tour.");
      setIsTourOpen(true);
    }
  }, []);

  const handleStartTraining = async () => {
    console.log("Starting training...");
    try {
      const response = await fetch("http://localhost:5000/train", {
        method: "POST",
      }); // Replace with your API endpoint
      if (response.ok) {
        console.log("Training started successfully.");
        setIsTrained(true);
        toast.success("Training started successfully!");
      } else {
        console.error("Failed to start training");
        toast.error("Failed to start training.");
      }
    } catch (error) {
      console.error("Error starting training:", error);
      toast.error("Error starting training.");
    }
  };

  const closeTour = () => {
    console.log("Closing tutorial tour.");
    setIsTourOpen(false);
    localStorage.setItem("hasCompletedTutorial", "true");
  };

  return (
    <div className={`${darkMode ? "dark" : ""} bg-mainBg min-h-screen`}>
      {/* Custom Tour */}
      <CustomTour
        steps={tutorialSteps}
        isOpen={isTourOpen}
        onClose={closeTour}
        darkMode={darkMode}
      />

      <div className="flex text-gray-100">
        {/* Sidebar */}
        <Sidebar />

        {/* Main Dashboard Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <Header />

          {/* Conditional Rendering Based on Training Status */}
          {isTrained === null ? (
            // Loading State
            <div className="flex justify-center items-center flex-1">
              <motion.div
                className="loader ease-linear rounded-full border-8 border-t-8 border-gray-200 h-32 w-32"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              ></motion.div>
            </div>
          ) : isTrained ? (
            // Dashboard Grid Content with 2 Columns per Row
            <main className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
              <div className="bg-sidebarBg p-6 rounded-lg shadow-lg">
                <h2 className="text-lg font-semibold mb-4">
                  Model Performance
                </h2>
                <LineChartComponent />
              </div>
              <div className="bg-sidebarBg p-6 rounded-lg shadow-lg">
                <MultiBarLineChartComponent />
              </div>
              <ShapExplanationChart />
              <div className="bg-sidebarBg p-6 rounded-lg shadow-lg">
                <UploadComponent />
              </div>
              <StackedBarChart />
              <HorizontalBarChart />
                <ModelMetricsCard />
                <ActualVsPredictedDynamic />
              <IntegratedGradientsView />
              <TrainingStatusTable />
              <LimeCharts />
            </main>
          ) : (
            // Start Training Component
            <main className="flex-1 p-4">
              <StartTrainingComponent onStart={handleStartTraining} />
            </main>
          )}
        </div>
      </div>

      {/* Toast Container */}
      <ToastContainer position="top-right" autoClose={5000} hideProgressBar />
    </div>
  );
};

export default DashboardPage;
