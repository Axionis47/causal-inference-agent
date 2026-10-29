import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell";
import Chat from "./pages/Chat";
import Landing from "./pages/Landing";
import DatasetDetails from "./pages/DatasetDetails";
import ContextReview from "./pages/ContextReview";
import NewDataset from "./pages/NewDataset";

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Landing />} />
        <Route path="/datasets/:datasetId" element={<DatasetDetails />} />
        <Route path="/datasets/:datasetId/reviews/:reviewId" element={<ContextReview />} />
        <Route path="/new" element={<NewDataset />} />
        <Route path="/d/:name" element={<Chat />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
