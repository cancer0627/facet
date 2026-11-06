import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { firstValueFrom } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';
import type { components } from '../../core/api/schema';

type BackgroundTask = components['schemas']['BackgroundTask'];
type TasksResponse = components['schemas']['BackgroundTasksResponse'];

@Component({
  selector: 'app-background-tasks-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DatePipe, MatButtonModule, MatDialogModule, MatIconModule, MatProgressBarModule, TranslatePipe],
  template: `
    <h2 mat-dialog-title class="flex items-center gap-2">
      <mat-icon>pending_actions</mat-icon>{{ 'tasks.title' | translate }}
    </h2>
    <mat-dialog-content class="!pt-2">
      <p class="text-sm opacity-70 mb-4">{{ 'tasks.description' | translate }}</p>
      @if (error()) {
        <p role="alert" class="mb-4 text-[var(--mat-sys-error)]">{{ 'tasks.load_error' | translate }}</p>
      }
      @if (actionError()) {
        <p role="alert" class="mb-4 text-[var(--mat-sys-error)]">{{ actionError() | translate }}</p>
      }
      @if (loading()) {
        <mat-progress-bar mode="indeterminate" />
      } @else if (!tasks().length && !error()) {
        <p class="py-8 text-center opacity-70">{{ 'tasks.empty' | translate }}</p>
      }
      @if (tasks().length) {
        <p class="text-sm mb-3" role="status">{{ 'tasks.active_count' | translate:{ count: activeCount() } }}</p>
      }
      <div class="flex flex-col gap-3">
        @for (task of tasks(); track task.id) {
          <article class="rounded-lg border border-[var(--mat-sys-outline-variant)] p-4 min-w-0">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
              <span class="font-medium">{{ kindLabel(task.kind) }}</span>
              <span class="text-sm" [style.color]="task.status === 'failed' ? 'var(--mat-sys-error)' : active(task) ? 'var(--mat-sys-primary)' : ''">
                {{ ('tasks.status.' + task.status) | translate }}
              </span>
            </div>
            @if (task.started_at) {
              <p class="text-xs opacity-60 mb-2">{{ task.started_at! * 1000 | date:'yyyy-MM-dd HH:mm:ss' }}</p>
            }
            @if (phaseLabel(task); as phase) {
              <p class="text-sm opacity-80 mb-2">{{ phase }}</p>
            }
            @if (active(task) || task.total) {
              <mat-progress-bar [mode]="task.total && task.current !== null && task.current !== undefined ? 'determinate' : 'indeterminate'"
                                [value]="percent(task)" />
            }
            @if (task.total !== null && task.total !== undefined && task.current !== null && task.current !== undefined) {
              <p class="text-sm mt-2">{{ 'tasks.progress' | translate:{ current: task.current!, total: task.total! } }}</p>
            } @else if (active(task)) {
              <p class="text-sm opacity-70 mt-2">{{ 'tasks.progress_unknown' | translate }}</p>
            }
            @if (task.succeeded !== null && task.succeeded !== undefined) {
              <p class="text-sm opacity-70 mt-2">{{ 'birds.counts' | translate:{ succeeded: task.succeeded!, skipped: task.skipped ?? 0, failed: task.failed ?? 0 } }}</p>
            }
            @if (task.error) {
              <p class="text-sm mt-2 break-words text-[var(--mat-sys-error)]">{{ task.error }}</p>
            }
            @if (task.can_terminate) {
              <div class="flex justify-end mt-2">
                <button mat-button [disabled]="!!terminating() || refreshing()" (click)="terminate(task)">
                  {{ (terminating() === task.id ? 'tasks.terminating' : 'tasks.terminate') | translate }}
                </button>
              </div>
            } @else if (active(task) && task.status !== 'stopping') {
              <p class="text-xs opacity-60 mt-2">{{ 'tasks.control_unavailable' | translate }}</p>
            }
          </article>
        }
      </div>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button [disabled]="refreshing() || !!terminating()" (click)="refresh()">{{ 'tasks.refresh' | translate }}</button>
      <button mat-button mat-dialog-close>{{ 'dialog.close' | translate }}</button>
    </mat-dialog-actions>
  `,
})
export class BackgroundTasksDialogComponent {
  private readonly api = inject(ApiService);
  private readonly i18n = inject(I18nService);
  protected readonly tasks = signal<BackgroundTask[]>([]);
  protected readonly loading = signal(true);
  protected readonly refreshing = signal(false);
  protected readonly error = signal(false);
  protected readonly actionError = signal('');
  protected readonly terminating = signal<string | null>(null);
  protected readonly activeCount = computed(() => this.tasks().filter(task => this.active(task)).length);
  private disposed = false;

  constructor() {
    inject(DestroyRef).onDestroy(() => {
      this.disposed = true;
    });
    void this.refresh();
  }

  protected active(task: BackgroundTask): boolean {
    return ['queued', 'running', 'stopping'].includes(task.status);
  }

  protected percent(task: BackgroundTask): number {
    return task.total ? Math.min(100, Math.max(0, (task.current ?? 0) / task.total * 100)) : 0;
  }

  protected kindLabel(kind: string): string {
    const key = 'tasks.kind.' + kind;
    const label = this.i18n.t(key);
    return label === key ? this.i18n.t('tasks.kind.maintenance') : label;
  }

  protected phaseLabel(task: BackgroundTask): string {
    if (!task.phase) return '';
    const key = (task.kind === 'elephants' ? 'elephants.phase.' : task.kind === 'birds' ? 'birds.phase.' : 'scan.phase.') + task.phase;
    const label = this.i18n.t(key);
    return label === key ? '' : label;
  }

  protected async refresh(): Promise<void> {
    if (this.refreshing() || this.terminating() || this.disposed) return;
    this.refreshing.set(true);
    try {
      const response = await firstValueFrom(this.api.get<TasksResponse>('/tasks'));
      if (this.disposed) return;
      this.tasks.set(response.tasks);
      this.error.set(false);
      this.actionError.set('');
    } catch {
      if (!this.disposed) this.error.set(true);
    } finally {
      if (!this.disposed) {
        this.loading.set(false);
        this.refreshing.set(false);
      }
    }
  }

  protected async terminate(task: BackgroundTask): Promise<void> {
    if (!task.can_terminate || this.terminating() || this.refreshing() || this.disposed) return;
    this.terminating.set(task.id);
    this.actionError.set('');
    try {
      const updated = await firstValueFrom(this.api.post<BackgroundTask>(
        '/tasks/' + encodeURIComponent(task.id) + '/terminate', { confirm: true },
      ));
      if (!this.disposed) this.tasks.update(tasks => tasks.map(item => item.id === task.id ? updated : item));
    } catch (err) {
      if (!this.disposed) this.actionError.set(err instanceof HttpErrorResponse && [403, 404, 409].includes(err.status)
        ? 'tasks.action_unavailable' : 'tasks.terminate_error');
    } finally {
      if (!this.disposed) this.terminating.set(null);
    }
  }
}
