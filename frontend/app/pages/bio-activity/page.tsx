// pages/biological-activity.js
"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header";
import BiologicalActivityTable from "@/app/components/BiologicalActivityTable";
import BiologicalActivityCharts from "@/app/components/BiologicalActivityCharts";
import ShapExplanationChartBIO from "@/app/components/ShapExplanationChartBIO";
import LimeExplanationChart from "@/app/components/LimeExplanationChart";
// Import your ChatComponent:
import ChatComponentBIO from "@/app/components/chatComponentBio";

import axios from "axios";

const BiologicalActivityPage = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Replace with your Flask backend URL
  const API_URL = "http://localhost:5000/api/check_biological_activity";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(API_URL);
        setData(response.data.results);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching biological activity data:", err);
        setError(true);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="bg-mainBg min-h-screen text-gray-100 flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <Header />

        {/* Content Area as a two-column flex */}
        <main className="flex-1 p-4 overflow-auto flex">
          {/* Left Column */}
          <div className="flex-1 mr-4">
            {loading ? (
              <div className="text-center text-xl">Loading data...</div>
            ) : error ? (
              <div className="text-center text-xl text-red-500">
                Error loading data. Please try again later.
              </div>
            ) : (
              <>
                {/* Table */}
                <BiologicalActivityTable data={data} />

                {/* Biological Activity Charts (existing) */}
                <BiologicalActivityCharts data={data} />

                {/* New SHAP Chart */}
                <ShapExplanationChartBIO />

                {/* New LIME Chart */}
                <LimeExplanationChart />
              </>
            )}
          </div>

          {/* Right Column: ChatComponent */}
          <div className="w-[400px] min-w-[300px] h-full">
            <ChatComponentBIO />
          </div>
        </main>
      </div>
    </div>
  );
};

export default BiologicalActivityPage;
