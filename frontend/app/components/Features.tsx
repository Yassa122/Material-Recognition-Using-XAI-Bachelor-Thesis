// components/Features.tsx
"use client";

import React from "react";
import { motion } from "framer-motion";
import ChemicalBond from "./ChemicalBond";
import { FaProjectDiagram, FaChartLine, FaBook } from "react-icons/fa"; // Importing icons from react-icons

const Features: React.FC = () => {
  // Animation variants for the container
  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  // Animation variants for each feature card
  const cardVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1 },
  };

  // Features data
  const features = [
    {
      icon: <FaProjectDiagram size={40} color="#ED8936" />, // Orange color
      title: "Molecular Visualization",
      description:
        "Interactive 3D models of chemical compounds for in-depth analysis.",
    },
    {
      icon: <FaChartLine size={40} color="#ED8936" />, // Orange color
      title: "Predictive Analytics",
      description:
        "AI-driven predictions for chemical reactions and compound behaviors.",
    },
    {
      icon: <FaBook size={40} color="#ED8936" />, // Orange color
      title: "Educational Resources",
      description:
        "Comprehensive guides and tutorials to enhance your chemistry knowledge.",
    },
  ];

  return (
    <section id="features" className="py-16 bg-gray-50 dark:bg-zinc-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        {/* Features Grid */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="bg-white dark:bg-zinc-700 p-8 rounded-lg shadow-lg flex flex-col items-center text-center transition-transform transform hover:scale-105"
              variants={cardVariants}
              whileHover={{ scale: 1.05 }}
              aria-label={feature.title}
            >
              {/* Icon */}
              <div className="mb-4">{feature.icon}</div>
              {/* Title */}
              <h3 className="text-2xl font-semibold text-gray-900 dark:text-zinc-100 mb-2">
                {feature.title}
              </h3>
              {/* Description */}
              <p className="text-gray-600 dark:text-zinc-400">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Chemical Bonds Connecting Features */}
        <ChemicalBond
          from={{ x: "33.333%", y: "0%" }}
          to={{ x: "33.333%", y: "100%" }}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
        <ChemicalBond
          from={{ x: "66.666%", y: "0%" }}
          to={{ x: "66.666%", y: "100%" }}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>
    </section>
  );
};

export default Features;
