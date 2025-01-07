// components/Testimonials.tsx
"use client";

import { motion } from "framer-motion";
import { FaQuoteLeft, FaQuoteRight } from "react-icons/fa";
import Image from "next/image"; // If using Next.js's Image component

interface Testimonial {
  quote: string;
  name: string;
  title: string;
  avatar: string; // URL to avatar image
}

const Testimonials: React.FC = () => {
  const testimonials: Testimonial[] = [
    {
      quote:
        "ExplainMat has transformed the way we visualize chemical structures. The AI predictions are incredibly accurate!",
      name: "Dr. Emily Chen",
      title: "Chemistry Professor, MIT",
      avatar: "/avatars/emily-chen.jpg", // Replace with actual image paths
    },
    {
      quote:
        "The interactive models make it so much easier to explain complex reactions to my students. Highly recommend!",
      name: "John Doe",
      title: "High School Chemistry Teacher",
      avatar: "/avatars/john-doe.jpg",
    },
    {
      quote:
        "As a researcher, the data analytics tools provided by ExplainMat have streamlined our workflow significantly.",
      name: "Dr. Michael Lee",
      title: "Research Scientist, Pfizer",
      avatar: "/avatars/michael-lee.jpg",
    },
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

  // Animation variants for each testimonial card
  const cardVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1 },
  };

  return (
    <section id="testimonials" className="py-16 bg-gray-50 dark:bg-zinc-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-zinc-100 mb-4">
            What Our Users Say
          </h2>
          <p className="text-lg sm:text-xl text-gray-600 dark:text-zinc-400">
            Hear from professionals who have transformed their workflow with
            ExplainMat.
          </p>
        </motion.div>

        {/* Testimonials Grid */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
        >
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              className="bg-white dark:bg-zinc-700 p-8 rounded-lg shadow-lg flex flex-col"
              variants={cardVariants}
              whileHover={{ scale: 1.02 }}
              aria-label={`Testimonial from ${testimonial.name}`}
            >
              {/* Quote Icon */}
              <div className="flex justify-start">
                <FaQuoteLeft size={24} className="text-orange-500" />
              </div>

              {/* Testimonial Text */}
              <p className="text-gray-700 dark:text-zinc-300 italic my-6 flex-1">
                "{testimonial.quote}"
              </p>

              {/* User Info */}
              <div className="flex items-center mt-4">
                {/* Avatar */}
                <div className="flex-shrink-0">
                  <Image
                    src={testimonial.avatar}
                    alt={`${testimonial.name} Avatar`}
                    width={50}
                    height={50}
                    className="rounded-full object-cover"
                  />
                </div>
                {/* Name and Title */}
                <div className="ml-4">
                  <p className="text-gray-900 dark:text-zinc-100 font-semibold">
                    {testimonial.name}
                  </p>
                  <p className="text-gray-600 dark:text-zinc-400 text-sm">
                    {testimonial.title}
                  </p>
                </div>
              </div>

              {/* Right Quote Icon */}
              <div className="flex justify-end mt-4">
                <FaQuoteRight size={24} className="text-orange-500" />
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Testimonials;
