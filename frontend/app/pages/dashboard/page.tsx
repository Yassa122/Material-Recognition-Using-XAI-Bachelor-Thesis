// dashboard/page.tsx
"use client";
import { useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import LineChartComponent from "@/app/components/LineChartComponent";
import UploadComponent from "@/app/components/UploadComponent";
import MultiBarLineChartComponent from "@/app/components/MultiBarLineChartComponent";
import ShapExplanationChart from "@/app/components/ShapExplanationChart";
import StackedBarChart from "@/app/components/StackedBarChart";
import HorizontalBarChart from "@/app/components/HorizontalBarChart";
import LimeCharts from "@/app/components/LimeCharts";
import TrainingStatusTable from "@/app/components/TrainingStatusTable";
import Header from "@/app/components/Header"; // Import the Header component

const DashboardPage = () => {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className={`${darkMode ? "dark" : ""} bg-mainBg min-h-screen`}>
      <div className="flex text-gray-100">
        {/* Sidebar with #202020 background color */}
        <Sidebar />

        {/* Main Dashboard Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <Header />

          {/* Toggle Button */}
          <div className="flex justify-end p-4">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="bg-gray-300 text-gray-800 p-2 rounded"
            >
              Toggle {darkMode ? "Light" : "Dark"} Mode
            </button>
          </div>

          {/* Dashboard Grid Content */}
          <main className="p-4 grid grid-cols-2 gap-4">
            <div className="bg-sidebarBg p-4 rounded-lg">
              <h2 className="text-lg font-semibold">Model Performance</h2>
              <LineChartComponent />
            </div>
            <div className="bg-sidebarBg p-4 rounded-lg">
              <MultiBarLineChartComponent />
            </div>
            <ShapExplanationChart />
            <div className="grid gap-4">
              <UploadComponent />
              <StackedBarChart />
            </div>
            <HorizontalBarChart />
            <LimeCharts />
            <TrainingStatusTable />
          </main>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
