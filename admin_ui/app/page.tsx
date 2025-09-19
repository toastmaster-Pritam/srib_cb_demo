"use client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import useWindowSize from "@rooks/use-window-size";

const campaignData = [
  { name: "Campaign A", ctr: 0.4 },
  { name: "Campaign B", ctr: 0.3 },
  {
    name: "Campaign C",
    ctr: 0.5,
  },
];

const analyticsData = [
  { day: "Mon", impressions: 400, clicks: 120 },
  { day: "Tue", impressions: 500, clicks: 180 },
  { day: "Wed", impressions: 700, clicks: 260 },
  { day: "Thu", impressions: 600, clicks: 200 },
  { day: "Fri", impressions: 800, clicks: 320 },
];

const deviceData = [
  { name: "Mobile", value: 55 },
  { name: "Tablet", value: 25 },
  { name: "Laptop", value: 20 },
];

const COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 },
};
export default function Home() {
  const { innerWidth } = useWindowSize();

  const isMobile = innerWidth !== null && innerWidth < 768;

  const chartHeight = isMobile ? 250 : 350;
  const pieRadius = isMobile ? 80 : 120;
  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8 overflow-x-hidden">
      <h1 className="text-2xl md:text-3xl font-bold">Admin Dashboard</h1>
      <motion.div
        className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4"
        initial="hidden"
        animate="visible"
        variants={{ visible: { transition: { staggerChildren: 0.2 } } }}
      >
        {[
          { title: "Total Campaigns", value: "12" },
          { title: "Total Users", value: "1,245" },
          { title: "Average CTR", value: "38%" },
        ].map((item, index) => (
          <motion.div key={index} variants={fadeInUp}>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg md:text-xl">
                  {item.title}
                </CardTitle>
              </CardHeader>

              <CardContent>
                <p className="text-xl md:text-2xl font-bold">{item.value}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        variants={fadeInUp}
        initial="hidden"
        animate="visible"
        transition={{ delay: 0.4 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg md:text-xl">
              CTR By Campaigns
            </CardTitle>
          </CardHeader>

          <CardContent>
            <ResponsiveContainer width={"100%"} height={chartHeight}>
              <BarChart
                data={campaignData}
                margin={{ top: 0, right: 30, left: 20, bottom: 10 }}
              >
                <XAxis dataKey={"name"} tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey={"ctr"} fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        variants={fadeInUp}
        initial="hidden"
        animate="visible"
        transition={{ delay: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg md:text-xl">
              Impression vs Click Over Time
            </CardTitle>
          </CardHeader>

          <CardContent>
            <ResponsiveContainer width={"100%"} height={chartHeight}>
              <LineChart data={analyticsData}>
                <XAxis dataKey={"day"} tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Line
                  type={"monotone"}
                  dataKey={"impressions"}
                  stroke="#8884d8"
                />
                <Line type={"monotone"} dataKey={"clicks"} stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        variants={fadeInUp}
        initial="hidden"
        animate="visible"
        transition={{ delay: 0.8 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg md:text-xl">
              CTR by Device Type
            </CardTitle>
          </CardHeader>

          <CardContent>
            <ResponsiveContainer width={"100%"} height={chartHeight}>
              <PieChart>
                <Pie
                  data={deviceData}
                  cx={"50%"}
                  cy={"50%"}
                  label
                  outerRadius={pieRadius}
                  dataKey={"value"}
                >
                  {deviceData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
