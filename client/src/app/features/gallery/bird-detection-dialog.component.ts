import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { firstValueFrom } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';

export interface BirdJob {
  id: string;
  status: string;
  phase: string;
  total: number;
  completed: number;
  succeeded: number;
  failed: number;
  skipped: number;
  cancel_requested: number;
  error?: string | null;
}

export interface BirdDialogResult {
  jobId: string;
  changed: boolean;
  viewResults: boolean;
}

const JOB_STORAGE_KEY = 'facet_bird_detection_job';

@Component({
  selector: 'app-bird-detection-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatDialogModule, MatButtonModule, MatCheckboxModule, MatProgressBarModule, TranslatePipe],
  template: `
    <h2 mat-dialog-title>{{ 'birds.title' | translate }}</h2>
    <mat-dialog-content class="!pt-2">
      @if (job(); as j) {
        <p class="mb-3">{{ ('birds.phase.' + j.phase) | translate }}</p>
        <mat-progress-bar [mode]="j.phase === 'loading_model' ? 'indeterminate' : 'determinate'"
                          [value]="j.total ? j.completed / j.total * 100 : 0" />
        <p class="mt-3">{{ 'birds.progress' | translate:{ completed: j.completed, total: j.total } }}</p>
        <p class="mt-2 text-sm opacity-70">{{ 'birds.counts' | translate:{ succeeded: j.succeeded, skipped: j.skipped, failed: j.failed } }}</p>
        @if (j.cancel_requested && active()) { <p class="mt-3">{{ 'birds.stopping' | translate }}</p> }
        @if (j.error) { <p role="alert" class="mt-3 text-[var(--mat-sys-error)]">{{ j.error }}</p> }
        <p class="text-sm opacity-70 mt-3">{{ 'birds.resume_hint' | translate }}</p>
      } @else {
        <p class="mb-3">{{ 'birds.description' | translate:{ count: data.paths.length } }}</p>
        <p class="text-sm opacity-70 mb-3">{{ 'birds.focus_score_hint' | translate }}</p>
        <label class="block mb-2">
          {{ 'birds.min_area' | translate }}
          <select class="ml-2 p-1 rounded bg-[var(--mat-sys-surface)]" [disabled]="busy()"
                  [value]="minAreaPercent()" (change)="setMinArea($event)">
            @for (option of areaOptions; track option.value) {
              <option [value]="option.value" [selected]="option.value === minAreaPercent()">{{ option.label | translate }}</option>
            }
          </select>
        </label>
        <p class="text-xs opacity-70 mb-3">{{ 'birds.min_area_hint' | translate }}</p>
        <mat-checkbox [checked]="tiled()" (change)="tiled.set($event.checked)" [disabled]="busy()">
          {{ 'birds.tiled' | translate }}
        </mat-checkbox>
        <p class="text-xs opacity-70 mb-2">{{ 'birds.tiled_hint' | translate }}</p>
        <mat-checkbox [checked]="recompute()" (change)="recompute.set($event.checked)" [disabled]="busy()">
          {{ 'birds.recompute' | translate }}
        </mat-checkbox>
      }
      @if (error()) { <p role="alert" class="mt-3 text-[var(--mat-sys-error)]">{{ error() | translate }}</p> }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      @if (job(); as j) {
        @if (active()) {
          <button mat-button [disabled]="!!j.cancel_requested || busy()" (click)="stop()">{{ 'birds.stop' | translate }}</button>
          <button mat-button (click)="finish(false)">{{ 'birds.background' | translate }}</button>
        } @else {
          <button mat-button (click)="reset()">{{ 'birds.run_again' | translate }}</button>
          <button mat-button (click)="finish(false)">{{ 'dialog.close' | translate }}</button>
          <button mat-flat-button [disabled]="!j.completed" (click)="finish(true)">{{ 'birds.view_results' | translate }}</button>
        }
      } @else {
        <button mat-button mat-dialog-close [disabled]="busy()">{{ 'ui.buttons.cancel' | translate }}</button>
        <button mat-flat-button [disabled]="busy() || !data.paths.length" (click)="start()">{{ 'birds.start' | translate }}</button>
      }
    </mat-dialog-actions>
  `,
})
export class BirdDetectionDialogComponent {
  private readonly api = inject(ApiService);
  private readonly ref = inject(MatDialogRef<BirdDetectionDialogComponent>);
  protected readonly data = inject<{ paths: string[] }>(MAT_DIALOG_DATA);
  protected readonly tiled = signal(true);
  protected readonly recompute = signal(false);
  protected readonly minAreaPercent = signal(1);
  protected readonly areaOptions = [
    { value: 0, label: 'birds.area_all' }, { value: .25, label: 'birds.area_small' },
    { value: 1, label: 'birds.area_medium' }, { value: 3, label: 'birds.area_large' },
    { value: 7, label: 'birds.area_xlarge' },
  ];
  protected readonly busy = signal(false);
  protected readonly job = signal<BirdJob | null>(null);
  protected readonly error = signal('');
  protected readonly active = computed(() => !!this.job() && !['completed', 'cancelled', 'failed'].includes(this.job()!.status));
  private timer?: ReturnType<typeof setTimeout>;
  private disposed = false;

  constructor() {
    inject(DestroyRef).onDestroy(() => {
      this.disposed = true;
      clearTimeout(this.timer);
    });
    const existing = sessionStorage.getItem(JOB_STORAGE_KEY);
    if (existing) {
      this.busy.set(true);
      void this.poll(existing).finally(() => this.busy.set(false));
    }
  }

  protected async start(): Promise<void> {
    if (this.busy() || !this.data.paths.length) return;
    this.busy.set(true);
    this.ref.disableClose = true;
    this.error.set('');
    try {
      const job = await firstValueFrom(this.api.post<BirdJob>('/birds/jobs', {
        paths: this.data.paths, tiled: this.tiled(), recompute: this.recompute(),
        min_area_percent: this.minAreaPercent(),
      }));
      sessionStorage.setItem(JOB_STORAGE_KEY, job.id);
      this.job.set(job);
      await this.poll(job.id);
    } catch (err) {
      this.error.set(err instanceof HttpErrorResponse && err.status === 409 ? 'birds.busy' : 'birds.start_error');
    } finally {
      this.busy.set(false);
      this.ref.disableClose = false;
    }
  }

  protected setMinArea(event: Event): void {
    const value = Number((event.target as HTMLSelectElement).value);
    if (this.areaOptions.some(option => option.value === value)) this.minAreaPercent.set(value);
  }

  private async poll(id: string): Promise<void> {
    try {
      const job = await firstValueFrom(this.api.get<BirdJob>('/birds/jobs/' + id));
      if (this.disposed) return;
      this.job.set(job);
      this.error.set('');
      if (this.active()) this.timer = setTimeout(() => void this.poll(id), 1500);
    } catch (err) {
      if (this.disposed) return;
      if (err instanceof HttpErrorResponse && err.status === 404) {
        sessionStorage.removeItem(JOB_STORAGE_KEY);
        this.job.set(null);
      } else {
        this.error.set('birds.progress_error');
        this.timer = setTimeout(() => void this.poll(id), 3000);
      }
    }
  }

  protected async stop(): Promise<void> {
    const job = this.job();
    if (!job || this.busy()) return;
    this.busy.set(true);
    try {
      this.job.set(await firstValueFrom(this.api.post<BirdJob>('/birds/jobs/' + job.id + '/cancel', { confirm: true })));
    } catch {
      this.error.set('birds.progress_error');
    } finally {
      this.busy.set(false);
    }
  }

  protected reset(): void {
    clearTimeout(this.timer);
    sessionStorage.removeItem(JOB_STORAGE_KEY);
    this.job.set(null);
    this.error.set('');
  }

  protected finish(viewResults: boolean): void {
    const job = this.job();
    if (job) this.ref.close({ jobId: job.id, changed: job.completed > 0, viewResults } satisfies BirdDialogResult);
  }
}
