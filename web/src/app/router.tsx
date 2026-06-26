import { lazy, Suspense } from "react";
import {
  createRootRoute,
  createRoute,
  createRouter,
  redirect,
  retainSearchParams,
  Outlet,
} from "@tanstack/react-router";
import { z } from "zod";
import { client, queryClient } from "@/lib/api/client";
import { Shell } from "./shell";
import { ErrorMessage, Loading } from "@/components/feedback";
const LoginPage = lazy(() =>
  import("@/features/auth/login").then((module) => ({
    default: module.LoginPage,
  })),
);
const DatasetLibrary = lazy(() =>
  import("@/features/datasets/library").then((module) => ({
    default: module.DatasetLibrary,
  })),
);
const DatasetDetail = lazy(() =>
  import("@/features/datasets/detail").then((module) => ({
    default: module.DatasetDetail,
  })),
);
const ContextWorkspace = lazy(() =>
  import("@/features/context/workspace").then((module) => ({
    default: module.ContextWorkspace,
  })),
);
const ChatWorkspace = lazy(() =>
  import("./conversation-page").then((module) => ({
    default: module.ConversationPage,
  })),
);

const root = createRootRoute({
  component: () => (
    <Suspense fallback={<Loading />}>
      <Outlet />
    </Suspense>
  ),
  notFoundComponent: () => (
    <main className="p-10">
      <h1>Page not found</h1>
      <a href="/datasets" className="text-primary">
        Return to dataset library
      </a>
    </main>
  ),
  errorComponent: ({ error }) => (
    <main className="p-10">
      <ErrorMessage error={error} />
      <a href="/datasets" className="text-primary">
        Return to dataset library
      </a>
    </main>
  ),
});
const index = createRoute({
  getParentRoute: () => root,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/datasets", search: { page: 1, search: "" } });
  },
});
const login = createRoute({
  getParentRoute: () => root,
  path: "/login",
  component: LoginPage,
});
const protectedRoute = createRoute({
  getParentRoute: () => root,
  id: "_authenticated",
  component: Shell,
  beforeLoad: async () => {
    await queryClient.fetchQuery({
      queryKey: ["get", "/api/auth/me"],
      queryFn: async () => {
        const { data, error, response } = await client.GET("/api/auth/me");
        if (response.status === 401) throw redirect({ to: "/login" });
        if (error) throw error;
        return data;
      },
    });
  },
});
const pageSchema = z.coerce.number().int().min(1).max(100_000).catch(1);
const library = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/datasets",
  validateSearch: z.object({
    page: pageSchema,
    search: z.string().max(200).catch(""),
  }),
  component: () => <DatasetLibrary {...library.useSearch()} />,
});
const detail = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/datasets/$datasetId",
  validateSearch: z.object({
    page: pageSchema,
    fromConversation: z.string().uuid().optional().catch(undefined),
  }),
  search: { middlewares: [retainSearchParams(["fromConversation"])] },
  component: () => (
    <DatasetDetail
      datasetId={detail.useParams().datasetId}
      page={detail.useSearch().page}
    />
  ),
});
const context = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/datasets/$datasetId/context/$versionId",
  validateSearch: z.object({
    revision: z.string().uuid().optional().catch(undefined),
    fromConversation: z.string().uuid().optional().catch(undefined),
    section: z
      .string()
      .regex(/^(dataset|concepts|relationships|columns\.[a-f0-9]{32})$/)
      .catch("dataset"),
  }),
  search: { middlewares: [retainSearchParams(["fromConversation"])] },
  component: () => (
    <ContextWorkspace {...context.useParams()} {...context.useSearch()} />
  ),
});
const chat = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/datasets/$datasetId/chat/$versionId",
  validateSearch: z.object({
    conversation: z.string().uuid().optional().catch(undefined),
    revision: z.string().uuid().optional().catch(undefined),
    analysis: z.string().uuid().optional().catch(undefined),
    view: z.enum(["chat", "analysis"]).optional().catch(undefined),
  }),
  component: () => (
    <ChatWorkspace {...chat.useParams()} {...chat.useSearch()} />
  ),
});
export const router = createRouter({
  routeTree: root.addChildren([
    index,
    login,
    protectedRoute.addChildren([library, detail, context, chat]),
  ]),
  defaultPendingComponent: Loading,
});
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
