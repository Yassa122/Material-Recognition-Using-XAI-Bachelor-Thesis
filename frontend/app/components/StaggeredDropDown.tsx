// components/StaggeredDropDown.tsx

import { FiChevronDown } from "react-icons/fi"; // Only Feather Icons available in react-icons/fi
import { FaBrain, FaCogs } from "react-icons/fa"; // Font Awesome Icons
import { motion, AnimatePresence } from "framer-motion";
import { Dispatch, SetStateAction, useState, useRef, useEffect } from "react";
import { IconType } from "react-icons";

interface StaggeredDropDownProps {
  selectedModel: string;
  setSelectedModel: Dispatch<SetStateAction<string>>;
}

const StaggeredDropDown: React.FC<StaggeredDropDownProps> = ({
  selectedModel,
  setSelectedModel,
}) => {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const modelOptions = [
    {
      text: "Predictive Model",
      value: "predictive",
      Icon: FaBrain, // Using Font Awesome Brain Icon
    },
    {
      text: "Generative Model",
      value: "generative",
      Icon: FaCogs, // Using Font Awesome Cogs Icon
    },
  ];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      <motion.div animate={open ? "open" : "closed"}>
        <button
          onClick={() => setOpen((pv) => !pv)}
          className="flex items-center gap-2 px-6 py-3 rounded-md text-indigo-50 bg-neutral-800 hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-300"
          aria-haspopup="true"
          aria-expanded={open}
        >
          <span className="font-medium text-sm">{selectedModel}</span>
          <motion.span variants={iconVariants}>
            <FiChevronDown />
          </motion.span>
        </button>

        {/* Dropdown Menu */}
        <AnimatePresence>
          {open && (
            <motion.ul
              initial="closed"
              animate="open"
              exit="closed"
              variants={wrapperVariants}
              className="absolute right-0 mt-2 w-64 rounded-md bg-neutral-900 shadow-lg ring-1 ring-black ring-opacity-15 z-20 origin-top-right"
            >
              {modelOptions.map((option, index) => (
                <Option
                  key={option.value}
                  text={option.text}
                  Icon={option.Icon}
                  setOpen={setOpen}
                  onSelect={() => setSelectedModel(option.text)}
                />
              ))}
            </motion.ul>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

const Option = ({
  text,
  Icon,
  setOpen,
  onSelect,
}: {
  text: string;
  Icon: IconType;
  setOpen: Dispatch<SetStateAction<boolean>>;
  onSelect: () => void;
}) => {
  return (
    <motion.li
      variants={itemVariants}
      onClick={() => {
        onSelect();
        setOpen(false);
      }}
      className="flex items-center gap-3 px-6 py-3 text-sm text-gray-200  hover:text-indigo-700 cursor-pointer transition-colors"
    >
      <motion.span variants={actionIconVariants}>
        <Icon className="h-5 w-5 text-indigo-500" />
      </motion.span>
      <span>{text}</span>
    </motion.li>
  );
};

export default StaggeredDropDown;

const wrapperVariants = {
  open: {
    opacity: 1,
    scaleY: 1,
    transition: {
      when: "beforeChildren",
      staggerChildren: 0.05,
    },
  },
  closed: {
    opacity: 0,
    scaleY: 0,
    transition: {
      when: "afterChildren",
      staggerChildren: 0.05,
    },
  },
};

const iconVariants = {
  open: { rotate: 180 },
  closed: { rotate: 0 },
};

const itemVariants = {
  open: {
    opacity: 1,
    y: 0,
  },
  closed: {
    opacity: 0,
    y: -10,
  },
};

const actionIconVariants = {
  open: { scale: 1, y: 0 },
  closed: { scale: 0, y: -10 },
};
