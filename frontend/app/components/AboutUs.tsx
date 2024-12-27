// components/AboutUs.tsx
"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { FaBullseye, FaUsers } from "react-icons/fa"; // Importing icons from react-icons
import ChemicalBond from "./ChemicalBond";

const AboutUs: React.FC = () => {
  // Animation variants for the container
  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  // Animation variants for each content block
  const blockVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1 },
  };

  return (
    <section id="about-us" className="py-16 bg-gray-50 dark:bg-zinc-800">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-zinc-100 mb-4">
            About Us
          </h2>
          <p className="text-lg sm:text-xl text-gray-600 dark:text-zinc-400">
            ExplainMat is dedicated to revolutionizing the field of chemistry
            through innovative visualization and predictive tools powered by
            artificial intelligence.
          </p>
        </motion.div>

        {/* Content Blocks */}
        <motion.div
          className="flex flex-col md:flex-row items-center md:space-x-8"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
        >
          {/* Image Section */}
          <motion.div
            className="md:w-1/2 mb-8 md:mb-0 relative"
            variants={blockVariants}
          >
            <div className="overflow-hidden rounded-lg shadow-lg">
              <Image
                src="/about-us.jpg" // Ensure you have this image in your public folder
                alt="About ExplainMat"
                width={600}
                height={400}
                className="w-full h-full object-cover transform hover:scale-105 transition-transform duration-500"
              />
              {/* Gradient Overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent"></div>
            </div>
          </motion.div>

          {/* Text Section */}
          <motion.div className="md:w-1/2" variants={blockVariants}>
            {/* Our Mission */}
            <div className="mb-8">
              <div className="flex items-center mb-4">
                <FaBullseye size={24} className="text-orange-500 mr-2" />
                <h3 className="text-2xl font-semibold text-gray-900 dark:text-zinc-100">
                  Our Mission
                </h3>
              </div>
              <p className="text-gray-600 dark:text-zinc-400">
                To empower chemists and researchers with state-of-the-art tools
                that simplify complex processes, enhance accuracy, and foster
                innovation in chemical research and education.
              </p>
            </div>

            {/* Our Team */}
            <div>
              <div className="flex items-center mb-4">
                <FaUsers size={24} className="text-orange-500 mr-2" />
                <h3 className="text-2xl font-semibold text-gray-900 dark:text-zinc-100">
                  Our Team
                </h3>
              </div>
              <p className="text-gray-600 dark:text-zinc-400">
                A diverse team of chemists, data scientists, and software
                engineers committed to pushing the boundaries of what's possible
                in chemical visualization and AI-driven prediction.
              </p>
            </div>
          </motion.div>
        </motion.div>

        {/* Chemical Bonds Decoration */}
        <div className="relative mt-12">
          <ChemicalBond
            from={{ x: "0%", y: "50%" }}
            to={{ x: "100%", y: "50%" }}
            className="w-full h-1 absolute top-1/2 left-0 pointer-events-none"
          />
        </div>
      </div>
    </section>
  );
};

export default AboutUs;
