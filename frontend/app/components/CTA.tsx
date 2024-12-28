// components/CTA.tsx
"use client";

import React from "react";
import { motion } from "framer-motion";
import ChemicalReaction from "./ChemicalReaction";
import { FaReact, FaBolt, FaFire } from "react-icons/fa"; // Example icons

const CTA: React.FC = () => {
  // Example chemical reactions
  const reactions = [
    {
      reaction: "Combustion of Methane",
      equation: "CH₄ + 2 O₂ → CO₂ + 2 H₂O",
      icon: <FaFire size={24} className="text-orange-500" />,
    },
    {
      reaction: "Photosynthesis",
      equation: "6 CO₂ + 6 H₂O + light → C₆H₁₂O₆ + 6 O₂",
      icon: <FaBolt size={24} className="text-orange-500" />,
    },
    {
      reaction: "Formation of Water",
      equation: "2 H₂ + O₂ → 2 H₂O",
      icon: <FaReact size={24} className="text-orange-500" />,
    },
    // Add more reactions as desired
  ];

  // Animation variants for the container
  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  // Animation variants for each reaction card
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <section id="cta" className="py-16 bg-neutral-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative">
        {/* Background Decoration */}
        <div className="absolute inset-0 -z-10">
          <div className="bg-gradient-to-tr from-orange-500 via-yellow-400 to-orange-500 opacity-50 h-full w-full"></div>
        </div>

        {/* Heading */}
        <motion.h2
          className="text-3xl sm:text-4xl font-extrabold text-white mb-8 relative z-10"
          initial={{ opacity: 0, y: -20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Discover the Power of Chemical Reactions
        </motion.h2>

        {/* Reactions Grid */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-8 justify-center relative z-10"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
        >
          {reactions.map((reaction, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              className="flex justify-center"
            >
              <ChemicalReaction
                reaction={reaction.reaction}
                equation={reaction.equation}
              />
            </motion.div>
          ))}
        </motion.div>

        {/* Optional Call to Action Button */}
        <motion.div
          className="mt-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <a
            href="#get-started"
            className="inline-block px-8 py-3 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500 shadow-md"
            aria-label="Get Started"
          >
            Get Started
          </a>
        </motion.div>
      </div>
    </section>
  );
};

export default CTA;
