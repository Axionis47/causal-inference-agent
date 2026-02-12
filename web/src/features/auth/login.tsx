import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "@tanstack/react-router";
import { ArrowRight, LockKeyhole, LoaderCircle } from "lucide-react";
import { api, queryClient } from "@/lib/api/client";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ErrorMessage } from "@/components/feedback";

const schema = z.object({
  email: z.email("Enter a valid email address."),
  password: z.string().min(1, "Enter your password."),
});
type Values = z.infer<typeof schema>;

export function LoginPage() {
  const navigate = useNavigate();
  const form = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: { email: "", password: "" },
  });
  const login = api.useMutation("post", "/api/auth/login", {
    onSuccess: async () => {
      queryClient.clear();
      await navigate({ to: "/datasets", search: { page: 1, search: "" } });
    },
  });
  return (
    <main className="login-page">
      <section className="login-story" aria-label="About the workspace">
        <Brand />
        <div className="login-story-copy">
          <div className="eyebrow">A workspace for causal inquiry</div>
          <h2 className="login-title">
            Good questions.
            <br />
            <span>Grounded in data.</span>
          </h2>
          <p className="subtext max-w-sm">
            Keep your source data intact.
            <br />
            Inspect the evidence, one dataset at a time.
          </p>
          <div className="terminal-art" aria-hidden="true">
            <div className="text-muted-foreground">// the starting point</div>
            <div>source.csv</div>
            <div>&nbsp; └─ saved version</div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp; └─ rows + columns</div>
          </div>
        </div>
        <div className="login-story-copy eyebrow">CAUSAL / DATA WORKSPACE</div>
      </section>
      <section className="login-form-wrap" aria-labelledby="login-heading">
        <div className="login-form">
          <LockKeyhole size={22} className="text-primary" aria-hidden="true" />
          <h1 id="login-heading">Log in to your workspace</h1>
          <p className="subtext">Your datasets, right where you left them.</p>
          <form
            onSubmit={form.handleSubmit((body) => login.mutate({ body }))}
            noValidate
          >
            <div className="form-field">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                type="email"
                spellCheck={false}
                autoComplete="username"
                placeholder="you@example.com"
                {...form.register("email")}
                aria-invalid={!!form.formState.errors.email}
                aria-describedby={
                  form.formState.errors.email ? "email-error" : undefined
                }
              />
              {form.formState.errors.email ? (
                <span id="email-error" className="field-error">
                  {form.formState.errors.email.message}
                </span>
              ) : null}
            </div>
            <div className="form-field">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                {...form.register("password")}
                aria-invalid={!!form.formState.errors.password}
                aria-describedby={
                  form.formState.errors.password ? "password-error" : undefined
                }
              />
              {form.formState.errors.password ? (
                <span id="password-error" className="field-error">
                  {form.formState.errors.password.message}
                </span>
              ) : null}
            </div>
            {login.error ? <ErrorMessage error={login.error} /> : null}
            <Button
              type="submit"
              className="w-full mt-6"
              size="lg"
              disabled={login.isPending}
            >
              {login.isPending ? (
                <LoaderCircle className="animate-spin" aria-hidden="true" />
              ) : null}
              {login.isPending ? "Logging in..." : "Log in"}
              <ArrowRight size={16} aria-hidden="true" />
            </Button>
          </form>
          <div className="login-footer">
            Private access. Use the account provided by your workspace owner.
          </div>
        </div>
      </section>
    </main>
  );
}
