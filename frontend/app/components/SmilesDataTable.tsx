// components/SmilesDataTable.tsx
import { useState } from "react";
import { FaEllipsisV, FaChevronLeft, FaChevronRight } from "react-icons/fa";

interface SMILESDatum {
  name: string;
  smiles: string;
  molecularWeight?: string;
  meltingPoint?: string;
  dateAdded?: string;
  [key: string]: any; // For dynamic properties
}

interface SmilesDataTableProps {
  data: SMILESDatum[];
}

const SmilesDataTable: React.FC<SmilesDataTableProps> = ({ data }) => {
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 5;

  // Calculate total pages
  const totalPages = Math.ceil(data.length / itemsPerPage);

  // Get current page data
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = data.slice(indexOfFirstItem, indexOfLastItem);

  // Get table headers dynamically based on data keys
  const headers = data.length > 0 ? Object.keys(data[0]) : [];

  const handlePrevPage = () => {
    setCurrentPage((prev) => (prev === 1 ? prev : prev - 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => (prev === totalPages ? prev : prev + 1));
  };

  return (
    <div className="bg-[#202020] p-6 rounded-lg shadow-lg mt-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-white">SMILES Dataset</h2>
        <FaEllipsisV className="text-gray-400 cursor-pointer" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="text-gray-500 text-sm border-b border-gray-700">
              {headers.map((header) => (
                <th key={header} className="py-2">
                  {header.charAt(0).toUpperCase() + header.slice(1)}{" "}
                  <span className="text-xs">▼</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="text-gray-300 text-sm">
            {currentItems.map((compound, index) => (
              <tr key={index} className="border-b border-gray-700">
                {headers.map((header) => (
                  <td key={header} className="py-3">
                    {compound[header]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex justify-end items-center mt-4 space-x-2">
          <button
            onClick={handlePrevPage}
            disabled={currentPage === 1}
            className={`p-2 rounded-full ${
              currentPage === 1
                ? "bg-gray-700 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
            aria-label="Previous Page"
          >
            <FaChevronLeft className="text-white" />
          </button>
          <span className="text-gray-400">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={handleNextPage}
            disabled={currentPage === totalPages}
            className={`p-2 rounded-full ${
              currentPage === totalPages
                ? "bg-gray-700 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
            aria-label="Next Page"
          >
            <FaChevronRight className="text-white" />
          </button>
        </div>
      )}
    </div>
  );
};

export default SmilesDataTable;
