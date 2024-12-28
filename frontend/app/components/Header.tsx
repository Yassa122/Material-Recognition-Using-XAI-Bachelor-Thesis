// components/Header.tsx
"use client";
import {
  FaSearch,
  FaBell,
  FaInfoCircle,
  FaMoon,
  FaQuestionCircle,
} from "react-icons/fa";
import { MdOutlineAccountCircle } from "react-icons/md";

const Header = () => {
  const handleRestartTutorial = () => {
    localStorage.removeItem("hasCompletedTutorial");
    window.location.reload(); // Simple approach to restart the tutorial
  };

  return (
    <header className="flex items-center justify-between bg-zinc-900 p-4 rounded-lg shadow-lg">
      {/* Search Bar */}
      <div className="relative flex items-center bg-[#202020] rounded-full px-4 py-2 w-1/2 header-search">
        <FaSearch className="text-gray-400 mr-2" />
        <input
          type="text"
          placeholder="Search"
          className="bg-transparent text-gray-300 outline-none w-full"
        />
      </div>

      {/* Right Side - Icons and Profile */}
      <div className="flex items-center space-x-4">
        <FaBell className="text-gray-400 hover:text-white cursor-pointer header-notifications" />
        <FaMoon className="text-gray-400 hover:text-white cursor-pointer" />
        <FaInfoCircle className="text-gray-400 hover:text-white cursor-pointer" />
        {/* Restart Tutorial Icon */}
        <FaQuestionCircle
          className="text-gray-400 hover:text-white cursor-pointer"
          title="Restart Tutorial"
          onClick={handleRestartTutorial}
        />
        <MdOutlineAccountCircle className="text-gray-400 hover:text-white cursor-pointer rounded-full h-8 w-8 header-profile" />
      </div>
    </header>
  );
};

export default Header;
