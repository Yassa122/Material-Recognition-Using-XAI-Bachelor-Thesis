// components/Header.tsx
import { FaSearch, FaBell, FaInfoCircle, FaMoon } from "react-icons/fa";
import { MdOutlineAccountCircle } from "react-icons/md";
import { IoMdWallet } from "react-icons/io";

const Header = () => {
  return (
    <header className="flex items-center justify-between bg-zinc-900 p-4 rounded-lg shadow-lg">
      {/* Search Bar */}
      <div className="relative flex items-center bg-[#202020] rounded-full px-4 py-2 w-1/2">
        <FaSearch className="text-gray-400 mr-2" />
        <input
          type="text"
          placeholder="Search"
          className="bg-transparent text-gray-300 outline-none w-full"
        />
      </div>

      {/* Right Side - Balance, Icons, and Profile */}
      <div className="flex items-center space-x-4">
        {/* Ethereum Balance */}
        <div className="flex items-center bg-[#202020] text-white font-semibold px-4 py-2 rounded-full space-x-2">
          <IoMdWallet className="text-blue-400" />
          <span>1,924 ETH</span>
        </div>

        {/* Icons */}
        <FaBell className="text-gray-400 hover:text-white cursor-pointer" />
        <FaMoon className="text-gray-400 hover:text-white cursor-pointer" />
        <FaInfoCircle className="text-gray-400 hover:text-white cursor-pointer" />

        {/* Profile Picture */}
        <MdOutlineAccountCircle className="text-gray-400 hover:text-white cursor-pointer rounded-full h-8 w-8" />
      </div>
    </header>
  );
};

export default Header;
