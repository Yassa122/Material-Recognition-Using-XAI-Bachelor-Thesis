// components/Sidebar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image"; // Import for using SVG image
import StarsIcon from "@/public/stars.svg"; // Adjust path if needed
import {
  MdDashboard,
  MdCloudUpload,
  MdBarChart,
  MdLightbulb,
  MdSettings,
  MdPerson,
  MdLock,
} from "react-icons/md";

const SidebarLink = ({
  href,
  icon: Icon,
  label,
  customIcon, // Optional custom icon prop
  className, // Add a className prop
}: {
  href: string;
  icon?: React.ComponentType<{ className?: string }>;
  label: string;
  customIcon?: string;
  className?: string;
}) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <Link href={href}>
      <div
        className={`relative group flex items-center space-x-4 py-4 rounded-md ${
          isActive
            ? "bg-gray-800 text-blue-400"
            : "hover:bg-gray-700 text-gray-300"
        } ${className}`} // Apply the className prop
      >
        {isActive && (
          <div className="absolute right-0 w-1 h-full bg-blue-500 rounded-full"></div>
        )}

        {/* Use custom icon if provided; otherwise, use Icon component */}
        {customIcon ? (
          <Image
            src={customIcon}
            alt={`${label} Icon`}
            width={24}
            height={24}
            className={`group-hover:text-white ${
              isActive ? "text-blue-400" : "text-gray-400"
            }`}
          />
        ) : (
          Icon && (
            <Icon
              className={`h-6 w-6 ${
                isActive ? "text-blue-400" : "text-gray-400"
              } group-hover:text-white`}
            />
          )
        )}
        <span
          className={`text-base font-semibold ${
            isActive ? "text-blue-400" : "text-gray-300"
          } group-hover:text-white`}
        >
          {label}
        </span>
      </div>
    </Link>
  );
};

const Sidebar = () => {
  return (
    <div className="w-64 bg-sidebarBg text-gray-100 pl-6 min-h-screen">
      <h2 className="text-3xl font-bold py-8 text-center text-white">
        Explain Mat
      </h2>
      <hr className="border-gray-600 mb-8" />
      <nav className="space-y-4">
        <SidebarLink
          href="/pages/dashboard"
          icon={MdDashboard}
          label="Dashboard"
          className="sidebar-dashboard" // Add unique class
        />
        <SidebarLink
          href="/pages/data-upload"
          icon={MdCloudUpload}
          label="Data Upload"
          className="sidebar-data-upload"
        />
        <SidebarLink
          href="/pages/model-training"
          label="Model Training"
          customIcon={StarsIcon}
          className="sidebar-model-training"
        />
        <SidebarLink
          href="/pages/generative"
          icon={MdBarChart}
          label="Generated Molecules"
          className="sidebar-generated-molecules" // Optional: add if needed in tutorial
        />
        <SidebarLink
          href="/pages/predictions"
          icon={MdLightbulb}
          label="Predictions"
          className="sidebar-predictions" // Optional: add if needed
        />
        <SidebarLink
          href="/dashboard/settings"
          icon={MdSettings}
          label="Settings"
          className="sidebar-settings" // Optional: add if needed
        />
        <SidebarLink
          href="/dashboard/profile"
          icon={MdPerson}
          label="Profile"
          className="sidebar-profile" // Optional: add if needed
        />
        <SidebarLink
          href="/sign-in"
          icon={MdLock}
          label="Sign In"
          className="sidebar-sign-in" // Optional: add if needed
        />
      </nav>
    </div>
  );
};

export default Sidebar;
