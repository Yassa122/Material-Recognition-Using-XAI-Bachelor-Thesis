// pages/index.tsx
"use client";

import React, { Suspense } from "react";
import Navbar from "./components/Navbar";
import Features from "./components/Features";
import Testimonials from "./components/Testimonials";
import AboutUs from "./components/AboutUs";
import CTA from "./components/CTA";
import Contact from "./components/Contact";
import FloatingParticle from "./components/FloatingParticle";
import { motion } from "framer-motion";
import Link from "next/link";
import dynamic from "next/dynamic";

// Lazy load components for performance optimization
const AnimatedMolecule = dynamic(
  () => import("./components/AnimatedMolecule"),
  { suspense: true }
);
const ChemicalBond = dynamic(() => import("./components/ChemicalBond"), {
  suspense: true,
});
const ChemicalReaction = dynamic(
  () => import("./components/ChemicalReaction"),
  { suspense: true }
);
const RotatingAtoms = dynamic(() => import("./components/RotatingAtoms"), {
  suspense: true,
});

// Define animation variants outside the component for reusability
const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.3,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: { opacity: 1, y: 0 },
};

const Home: React.FC = () => {
  // Array of particles with different properties
  const particles = [
    {
      size: 8,
      color: "#F97316",
      initialPosition: { x: "20%", y: "30%" },
      hoverScale: 1.5,
    },
    {
      size: 6,
      color: "#3B82F6",
      initialPosition: { x: "70%", y: "60%" },
      hoverScale: 1.3,
    },
    {
      size: 10,
      color: "#06B6D4",
      initialPosition: { x: "40%", y: "80%" },
      hoverScale: 1.4,
    },
    {
      size: 7,
      color: "#34D399",
      initialPosition: { x: "85%", y: "20%" },
      hoverScale: 1.2,
    },
    // Add more particles as desired
  ];

  return (
    <div className="relative bg-gradient-to-b from-white to-gray-50 dark:from-zinc-900 dark:to-zinc-800 min-h-screen flex flex-col overflow-hidden">
      {/* Navbar */}
      <Navbar />

      {/* Floating Particles */}
      {particles.map((particle, index) => (
        <FloatingParticle
          key={index}
          size={particle.size}
          color={particle.color}
          initialPosition={particle.initialPosition}
          hoverScale={particle.hoverScale}
        />
      ))}

      {/* Hero Section */}
      <section className="relative flex flex-col justify-center items-center flex-1 p-8 sm:p-16 lg:p-24 text-center mt-16 md:mt-24">
        {/* Animated Molecule as Background */}
        <Suspense
          fallback={
            <div className="absolute inset-0 flex items-center justify-center">
              Loading...
            </div>
          }
        >
          {/* <AnimatedMolecule /> */}
        </Suspense>

        {/* Rotating Atoms */}
        <Suspense
          fallback={
            <div className="absolute inset-0 flex items-center justify-center">
              Loading Atoms...
            </div>
          }
        >
          <RotatingAtoms />
        </Suspense>

        {/* Content Container */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="relative z-10 max-w-3xl px-4"
        >
          <motion.h1
            variants={itemVariants}
            className="text-4xl sm:text-5xl lg:text-6xl font-extrabold bg-gradient-to-r from-orange-400 via-yellow-500 to-orange-600 text-transparent bg-clip-text leading-tight mb-6 animate-gradient relative z-10"
          >
            Transforming Chemical Insights with{" "}
            <span className="text-yellow-300">ExplainMat</span>
          </motion.h1>
          <motion.p
            variants={itemVariants}
            className="text-base sm:text-lg lg:text-xl text-gray-600 dark:text-zinc-400 mb-8 bg-gradient-to-r from-orange-400 via-gray-500 to-gray-600 text-transparent bg-clip-text bg-opacity-75 leading-relaxed tracking-wide"
          >
            Leveraging advanced AI to enhance chemical visualization and
            prediction for accurate and efficient scientific discoveries.
          </motion.p>
          <motion.div
            variants={itemVariants}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link
              href="#get-started"
              className="px-8 py-3 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-transform transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-orange-500 shadow-md"
              aria-label="Get Started"
            >
              Get Started
            </Link>
            <Link
              href="#learn-more"
              className="px-8 py-3 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-transform transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-gray-400 shadow-md"
              aria-label="Learn More"
            >
              Learn More
            </Link>
          </motion.div>
        </motion.div>

        {/* Chemical Bonds in Hero Section */}
        <Suspense
          fallback={
            <div className="absolute inset-0 flex items-center justify-center">
              Loading Bonds...
            </div>
          }
        >
          <ChemicalBond
            from={{ x: "25%", y: "50%" }}
            to={{ x: "75%", y: "50%" }}
          />
        </Suspense>

        {/* Optional: Add more ChemicalBond components as needed */}
      </section>

      {/* Features Section */}
      <Features />

      {/* Testimonials Section */}
      <Testimonials />

      {/* About Us Section */}
      <AboutUs />

      {/* Call-to-Action Section */}
      <CTA />

      {/* Contact Section */}
      <Contact />

      {/* Footer */}
      <footer className="bg-gray-50 dark:bg-zinc-900 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="flex justify-center space-x-6 mb-4">
            <Link
              href="https://twitter.com/yourprofile"
              className="text-gray-600 dark:text-zinc-400 hover:text-blue-500 transition"
              aria-label="Twitter"
            >
              {/* Twitter SVG Icon */}
              <svg
                className="w-6 h-6"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
                role="img"
              >
                {/* SVG Path */}
                <path d="M24 4.557a9.93 9.93 0 0 1-2.828.775 4.932 4.932 0 0 0 2.165-2.724c-.951.555-2.005.959-3.127 1.184a4.916 4.916 0 0 0-8.384 4.482A13.944 13.944 0 0 1 1.671 3.149a4.916 4.916 0 0 0 1.523 6.574A4.897 4.897 0 0 1 .964 9.1v.062a4.916 4.916 0 0 0 3.946 4.814 4.902 4.902 0 0 1-2.212.084 4.918 4.918 0 0 0 4.588 3.417A9.867 9.867 0 0 1 0 19.54a13.94 13.94 0 0 0 7.548 2.213c9.058 0 14.01-7.513 14.01-14.01 0-.213-.005-.425-.014-.636A10.025 10.025 0 0 0 24 4.557z" />
              </svg>
            </Link>
            <Link
              href="https://github.com/yourprofile"
              className="text-gray-600 dark:text-zinc-400 hover:text-blue-500 transition"
              aria-label="GitHub"
            >
              {/* GitHub SVG Icon */}
              <svg
                className="w-6 h-6"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
                role="img"
              >
                {/* SVG Path */}
                <path
                  fillRule="evenodd"
                  d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.73.083-.73 1.205.084 1.838 1.236 1.838 1.236 1.07 1.834 2.809 1.304 3.495.997.108-.775.42-1.305.763-1.605-2.665-.3-5.467-1.332-5.467-5.931 0-1.31.468-2.381 1.235-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.3 1.23a11.52 11.52 0 0 1 3-.405c1.02.005 2.045.138 3 .405 2.291-1.552 3.297-1.23 3.297-1.23.655 1.653.243 2.874.119 3.176.77.84 1.233 1.911 1.233 3.221 0 4.61-2.807 5.628-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .322.218.694.825.576C20.565 21.796 24 17.3 24 12c0-6.627-5.373-12-12-12z"
                  clipRule="evenodd"
                />
              </svg>
            </Link>
            {/* Add more social icons as needed */}
          </div>
          <p className="text-sm text-gray-600 dark:text-zinc-400">
            © {new Date().getFullYear()} ExplainMat. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
