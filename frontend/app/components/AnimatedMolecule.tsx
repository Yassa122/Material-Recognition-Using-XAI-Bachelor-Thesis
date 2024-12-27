// components/AnimatedMolecule.tsx
"use client";
import React from "react";
import { motion, Variants } from "framer-motion";

const moleculeVariants: Variants = {
  rotate: {
    rotate: 360,
    transition: {
      repeat: Infinity,
      repeatType: "loop",
      duration: 60, // Increased rotation speed for a dynamic effect
      ease: "linear",
    },
  },
  fadeIn: {
    opacity: [0, 1],
    transition: {
      duration: 1.5,
      ease: "easeInOut",
    },
  },
};

const orbitVariants: Variants = {
  orbit: {
    rotate: 360,
    transition: {
      repeat: Infinity,
      repeatType: "loop",
      duration: 40,
      ease: "linear",
    },
  },
};

const electronVariants: Variants = {
  orbit1: {
    rotate: 360,
    transition: {
      repeat: Infinity,
      repeatType: "loop",
      duration: 6,
      ease: "linear",
    },
  },
  orbit2: {
    rotate: -360,
    transition: {
      repeat: Infinity,
      repeatType: "loop",
      duration: 9,
      ease: "linear",
    },
  },
  orbit3: {
    rotate: 360,
    transition: {
      repeat: Infinity,
      repeatType: "loop",
      duration: 7,
      ease: "linear",
    },
  },
};

const AnimatedMolecule: React.FC = () => {
  return (
    <motion.div
      className="absolute top-0 left-0 w-full h-full flex justify-center items-center pointer-events-none"
      variants={moleculeVariants}
      animate={["rotate", "fadeIn"]}
      aria-label="Animated Molecule Visualization"
    >
      {/* Professional Molecule SVG */}
      <svg
        className="w-72 h-72 sm:w-96 sm:h-96 lg:w-[600px] lg:h-[600px] opacity-90"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Definitions for gradients and filters */}
        <defs>
          {/* Radial Gradient for the Nucleus */}
          <radialGradient id="nucleusGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FFD700" /> {/* Gold */}
            <stop offset="100%" stopColor="#FFA500" /> {/* Orange */}
          </radialGradient>

          {/* Radial Gradients for Orbits */}
          <radialGradient id="orbitGradient1" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#6A5ACD" stopOpacity="0.7" />{" "}
            {/* SlateBlue */}
            <stop offset="100%" stopColor="#483D8B" stopOpacity="0.7" />{" "}
            {/* DarkSlateBlue */}
          </radialGradient>
          <radialGradient id="orbitGradient2" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FF6347" stopOpacity="0.7" />{" "}
            {/* Tomato */}
            <stop offset="100%" stopColor="#FF4500" stopOpacity="0.7" />{" "}
            {/* OrangeRed */}
          </radialGradient>
          <radialGradient id="orbitGradient3" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#32CD32" stopOpacity="0.7" />{" "}
            {/* LimeGreen */}
            <stop offset="100%" stopColor="#228B22" stopOpacity="0.7" />{" "}
            {/* ForestGreen */}
          </radialGradient>

          {/* Glow Filter */}
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#FFFFFF" />
          </filter>

          {/* Highlight Filter for Hover Effects */}
          <filter
            id="highlightFilter"
            x="-50%"
            y="-50%"
            width="200%"
            height="200%"
          >
            <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#FFFFFF" />
          </filter>
        </defs>

        {/* Outer Orbit Path */}
        <motion.circle
          cx="100"
          cy="100"
          r="85"
          stroke="url(#orbitGradient3)"
          strokeWidth="2"
          fill="none"
          strokeDasharray="15,15"
          opacity="0.8"
          variants={orbitVariants}
          animate="orbit"
        />

        {/* Middle Orbit Path */}
        <motion.circle
          cx="100"
          cy="100"
          r="55"
          stroke="url(#orbitGradient1)"
          strokeWidth="2"
          fill="none"
          strokeDasharray="15,15"
          opacity="0.8"
          variants={orbitVariants}
          animate="orbit"
        />

        {/* Inner Orbit Path */}
        <motion.circle
          cx="100"
          cy="100"
          r="25"
          stroke="url(#orbitGradient2)"
          strokeWidth="2"
          fill="none"
          strokeDasharray="15,15"
          opacity="0.8"
          variants={orbitVariants}
          animate="orbit"
        />

        {/* Atom Core */}
        <motion.circle
          cx="100"
          cy="100"
          r="20"
          stroke="url(#nucleusGradient)"
          strokeWidth="4"
          fill="url(#nucleusGradient)"
          filter="url(#glow)"
          whileHover={{
            scale: 1.05,
            filter: "url(#highlightFilter)",
            transition: { duration: 0.3 },
          }}
          aria-label="Atom Core"
        />

        {/* Electrons on Orbit 1 */}
        <g>
          {[...Array(8)].map((_, index) => (
            <motion.circle
              key={`orbit1-electron-${index}`}
              cx="100"
              cy="15"
              r="4"
              fill="#6A5ACD" // SlateBlue
              variants={electronVariants.orbit1}
              animate="orbit1"
              style={{
                originX: "100px",
                originY: "100px",
              }}
              whileHover={{
                scale: 1.6,
                fill: "#8470FF", // LightSlateBlue
                filter: "url(#highlightFilter)",
                transition: { duration: 0.3 },
              }}
              aria-label={`Electron ${index + 1}`}
            >
              {/* Tooltip for Accessibility */}
              <title>Electron {index + 1} - Energy Level 1</title>
            </motion.circle>
          ))}
        </g>

        {/* Electrons on Orbit 2 */}
        <g>
          {[...Array(8)].map((_, index) => (
            <motion.circle
              key={`orbit2-electron-${index}`}
              cx="100"
              cy="70"
              r="4"
              fill="#FF6347" // Tomato
              variants={electronVariants.orbit2}
              animate="orbit2"
              style={{
                originX: "100px",
                originY: "100px",
              }}
              whileHover={{
                scale: 1.6,
                fill: "#FF7F50", // Coral
                filter: "url(#highlightFilter)",
                transition: { duration: 0.3 },
              }}
              aria-label={`Electron ${index + 9}`}
            >
              {/* Tooltip for Accessibility */}
              <title>Electron {index + 9} - Energy Level 2</title>
            </motion.circle>
          ))}
        </g>

        {/* Electrons on Orbit 3 */}
        <g>
          {[...Array(8)].map((_, index) => (
            <motion.circle
              key={`orbit3-electron-${index}`}
              cx="100"
              cy="130"
              r="4"
              fill="#32CD32" // LimeGreen
              variants={electronVariants.orbit3}
              animate="orbit3"
              style={{
                originX: "100px",
                originY: "100px",
              }}
              whileHover={{
                scale: 1.6,
                fill: "#3CB371", // MediumSeaGreen
                filter: "url(#highlightFilter)",
                transition: { duration: 0.3 },
              }}
              aria-label={`Electron ${index + 17}`}
            >
              {/* Tooltip for Accessibility */}
              <title>Electron {index + 17} - Energy Level 3</title>
            </motion.circle>
          ))}
        </g>
      </svg>
    </motion.div>
  );
};

export default AnimatedMolecule;
