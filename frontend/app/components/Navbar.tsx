"use client";
import Link from "next/link";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaBars, FaTimes, FaSun, FaMoon } from "react-icons/fa";

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [isDark, setIsDark] = useState<boolean>(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const toggleDarkMode = () => {
    setIsDark(!isDark);
  };

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  useEffect(() => {
    // Check local storage for theme preference
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme === "dark") {
      setIsDark(true);
    } else {
      setIsDark(false);
    }
  }, []);

  return (
    <nav className="w-full bg-white dark:bg-zinc-800 backdrop-blur-3xl shadow fixed top-0 left-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0 flex items-center">
            <Link
              href="/"
              className="text-2xl font-bold text-indigo-600 hover:text-indigo-700 transition"
            >
              ExplainMat
            </Link>
          </div>
          {/* Desktop Menu */}
          <div className="hidden md:flex space-x-6 items-center">
            <Link
              href="/features"
              className="text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
            >
              Features
            </Link>
            <Link
              href="/about"
              className="text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
            >
              About
            </Link>
            <Link
              href="/contact"
              className="text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
            >
              Contact
            </Link>
            <Link
              href="/pages/dashboard"
              className="ml-4 px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition shadow-md"
            >
              Get Started
            </Link>
            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="ml-4 text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded"
              aria-label="Toggle Dark Mode"
            >
              {isDark ? <FaSun size={20} /> : <FaMoon size={20} />}
            </button>
          </div>
          {/* Mobile Menu Button */}
          <div className="flex items-center md:hidden">
            <button
              onClick={toggleMenu}
              className="text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded"
              aria-label="Toggle Menu"
            >
              {isOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
            </button>
          </div>
        </div>
      </div>
      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden bg-white dark:bg-zinc-900 px-4 pt-4 pb-6 space-y-4 sm:px-6 overflow-hidden shadow-md"
          >
            <Link
              href="/features"
              className="block text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
              onClick={() => setIsOpen(false)} // Close menu on link click
            >
              Features
            </Link>
            <Link
              href="/about"
              className="block text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
              onClick={() => setIsOpen(false)}
            >
              About
            </Link>
            <Link
              href="/contact"
              className="block text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition"
              onClick={() => setIsOpen(false)}
            >
              Contact
            </Link>
            <Link
              href="/pages/dashboard"
              className="block px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition shadow-md"
              onClick={() => setIsOpen(false)}
            >
              Get Started
            </Link>
            {/* Dark Mode Toggle in Mobile Menu */}
            <button
              onClick={toggleDarkMode}
              className="flex items-center text-zinc-600 dark:text-zinc-100 hover:text-indigo-600 transition focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded px-2 py-1"
              aria-label="Toggle Dark Mode"
            >
              {isDark ? <FaSun size={20} /> : <FaMoon size={20} />}
              <span className="ml-2 text-sm">
                {isDark ? "Light Mode" : "Dark Mode"}
              </span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

export default Navbar;
