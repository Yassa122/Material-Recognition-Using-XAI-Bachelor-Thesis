"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";

const PredictedPropertiesTable = () => {
  const [predictedData, setPredictedData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1); // Track the current page
  const [totalPages, setTotalPages] = useState(1); // Track the total number of pages

  const limit = 5; // Set the number of records per page

  // Function to fetch the data for the current page
  const fetchPredictedData = async (page) => {
    setLoading(true); // Set loading to true before fetching data
    try {
      const response = await fetch(
        `http://localhost:5000/api/predictions?page=${page}&limit=${limit}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch predictions");
      }
      const data = await response.json();
      setPredictedData(data.predictions); // Set the fetched data into state
      setTotalPages(data.pagination.total_pages); // Set total pages for pagination
    } catch (error) {
      console.error("Error fetching predicted data:", error);
    } finally {
      setLoading(false); // Set loading to false once the data is fetched
    }
  };

  // Fetch data when the component mounts or when the page number changes
  useEffect(() => {
    fetchPredictedData(page);
  }, [page]); // Dependency on `page` to trigger re-fetching when it changes

  // Helper function to handle the transition effect for smooth page change
  const handlePageChange = (newPage) => {
    setPage(newPage);
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="bg-zinc-900 p-6 rounded-xl shadow-lg mt-6">
      <h3 className="text-lg font-bold text-gray-200 mb-4">
        Predicted Properties
      </h3>
      <table className="w-full text-left border-collapse border border-gray-700">
        <thead className="bg-zinc-800 text-gray-400">
          <tr>
            <th className="p-4 border border-gray-700">#</th>
            <th className="p-4 border border-gray-700">SMILES</th>
            <th className="p-4 border border-gray-700">Predicted pIC50</th>
            <th className="p-4 border border-gray-700">Predicted logP</th>
            <th className="p-4 border border-gray-700">Predicted Num Atoms</th>
          </tr>
        </thead>
        <AnimatePresence>
          <motion.tbody
            key={page} // This ensures that we animate the rows when the page changes
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {predictedData.map((item, index) => (
              <motion.tr
                key={item.SMILES}
                className={`${
                  index % 2 === 0 ? "bg-zinc-800" : "bg-zinc-700"
                } hover:bg-zinc-600`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <td className="p-4 border border-gray-700 text-gray-300">
                  {index + 1}
                </td>
                <td className="p-4 border border-gray-700 text-blue-400 font-mono">
                  <Link
                    href={`/pages/compound/${encodeURIComponent(item.SMILES)}`}
                    className="hover:underline"
                  >
                    {item.SMILES}
                  </Link>
                </td>
                <td className="p-4 border border-gray-700 text-green-400 font-medium">
                  {item.Predicted_pIC50}
                </td>
                <td className="p-4 border border-gray-700 text-yellow-400 font-medium">
                  {item.Predicted_logP}
                </td>
                <td className="p-4 border border-gray-700 text-purple-400 font-medium">
                  {item.Predicted_num_atoms}
                </td>
              </motion.tr>
            ))}
          </motion.tbody>
        </AnimatePresence>
      </table>

      {/* Pagination Controls */}
      <div className="mt-4 flex justify-between items-center">
        <button
          onClick={() => handlePageChange(Math.max(page - 1, 1))}
          disabled={page === 1}
          className={`bg-blue-500 text-white px-4 py-2 rounded-lg ${
            page === 1 ? "cursor-not-allowed opacity-50" : ""
          }`}
        >
          Previous
        </button>
        <span className="text-gray-200">
          Page {page} of {totalPages}
        </span>
        <button
          onClick={() => handlePageChange(Math.min(page + 1, totalPages))}
          disabled={page === totalPages}
          className={`bg-blue-500 text-white px-4 py-2 rounded-lg ${
            page === totalPages ? "cursor-not-allowed opacity-50" : ""
          }`}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default PredictedPropertiesTable;
