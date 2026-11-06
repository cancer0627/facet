import { ChangeDetectionStrategy, Component, computed, effect, inject, input, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { ApiService } from '../../core/services/api.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';
import { FixedPipe } from '../../shared/pipes/fixed.pipe';

interface BirdResult {
  status: string | null;
  score: number | null;
  count: number | null;
  best_index: number | null;
  stale: boolean;
  outdated: boolean;
  min_area_percent?: number | null;
  excluded_small_count?: number | null;
  focus_unverified_count?: number | null;
  assessment?: 'pass' | 'fail' | 'review' | null;
  assessment_reason?: string | null;
  assessment_index?: number | null;
  focus_score?: number | null;
  focus_score_index?: number | null;
  error: string | null;
  elephants?: BirdResult['birds'];
  birds: { sharpness: number | null; confidence: number; width: number; height: number;
    reason?: string | null; area_percent?: number; exclusion_reason?: string | null; focus_verified?: boolean;
    assessment?: 'pass' | 'fail' | 'review'; assessment_reason?: string; focus_score?: number | null }[];
}

@Component({
  selector: 'app-bird-results',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TranslatePipe, FixedPipe, MatButtonModule],
  template: `
    @if (result(); as r) {
      @if (r.status) {
        <section class="border-t border-[var(--mat-sys-outline-variant)] pt-3 space-y-2">
          <h3 class="font-medium">{{ (labels() + '.title') | translate }}</h3>
          @if (!r.stale && !r.outdated && r.focus_score !== null && r.focus_score !== undefined) {
            <p class="font-medium">{{ (labels() + '.focus_score') | translate }}: {{ r.focus_score | fixed:1 }}</p>
          } @else {
            <p class="font-medium">{{ (labels() + '.assessment.review') | translate }}</p>
          }
          @if ((r.focus_score === null || r.focus_score === undefined || r.stale || r.outdated) && r.assessment_reason) {
            <p class="text-sm">{{ ((labels() + '.assessment_reason.') + r.assessment_reason) | translate }}</p>
          }
          <p class="text-xs opacity-70">{{ (labels() + '.focus_score_hint') | translate }}</p>
          @if (r.outdated) { <p>{{ (labels() + '.outdated') | translate }}</p> }
          @if (!r.outdated && r.focus_unverified_count) {
            <p class="text-sm">{{ (labels() + '.focus_unverified_count') | translate:{ count: r.focus_unverified_count } }}</p>
          }
          @if (!r.outdated && r.min_area_percent !== null && r.min_area_percent !== undefined) {
            <p class="text-xs opacity-70">{{ (labels() + '.size_rule') | translate:{ percent: r.min_area_percent } }}</p>
            @if (r.excluded_small_count) {
              <p class="text-xs opacity-70">{{ (labels() + '.small_excluded') | translate:{ count: r.excluded_small_count } }}</p>
            }
          }
          @if (r.stale) {
            <p>{{ (labels() + '.stale') | translate }}</p>
          } @else {
            @if (r.count) {
              <p>{{ (labels() + '.detected_count') | translate:{ count: r.count } }}</p>
              <button mat-button (click)="showPreview.set(!showPreview())">{{ (labels() + '.show_location') | translate }}</button>
              @if (showPreview()) {
                <img class="w-full rounded" [src]="preview()" [alt]="(labels() + '.location_alt') | translate" />
                <p class="text-xs opacity-70">{{ (labels() + '.box_hint') | translate }}</p>
                <label class="block text-sm">
                  {{ (labels() + '.inspect_bird') | translate }}
                  <select class="ml-2 p-1 rounded bg-[var(--mat-sys-surface)]"
                          [value]="selectedBird()" (change)="selectBird($event)">
                    @for (bird of r.birds; track $index) {
                      <option [value]="$index" [selected]="$index === selectedBird()">
                        {{ (labels() + '.bird_number') | translate:{ number: $index + 1 } }}
                        @if (!r.stale && !r.outdated && bird.focus_score !== null && bird.focus_score !== undefined) {
                          · {{ bird.focus_score | fixed:1 }}
                        } @else { · {{ (labels() + '.assessment.review') | translate }} }
                      </option>
                    }
                  </select>
                </label>
                @if (selectedBird() !== null) {
                  @if (r.birds[selectedBird()!]; as bird) {
                    @if (bird.area_percent !== undefined) {
                      <p class="text-sm">{{ (labels() + '.size_metric') | translate:{ width: bird.width, height: bird.height, percent: (bird.area_percent | fixed:3) } }}</p>
                    }
                    @if (bird.exclusion_reason) {
                      <p class="text-sm">{{ ((labels() + '.reason.') + bird.exclusion_reason) | translate }}</p>
                    }
                    @if ((bird.focus_score === null || bird.focus_score === undefined) && bird.assessment_reason) {
                      <p class="text-sm">{{ ((labels() + '.assessment_reason.') + bird.assessment_reason) | translate }}</p>
                    }
                  }
                  @if (!r.birds[selectedBird()!]?.assessment_reason && r.birds[selectedBird()!]?.reason && r.birds[selectedBird()!]?.reason !== r.birds[selectedBird()!]?.exclusion_reason) {
                    @let reason = r.birds[selectedBird()!]!.reason;
                    <p class="text-sm">{{ ((labels() + '.reason.') + reason) | translate }}</p>
                  }
                  <img class="w-full rounded" [src]="preview(selectedBird()!)" [alt]="(labels() + '.crop_alt') | translate" />
                }
              }
            }
          }
          @if (r.error) { <p role="alert">{{ r.error }}</p> }
        </section>
      }
    }
  `,
})
export class BirdResultsComponent {
  readonly path = input.required<string>();
  readonly species = input<'bird' | 'elephant'>('bird');
  protected readonly labels = computed(() => this.species() === 'elephant' ? 'elephants' : 'birds');
  private readonly api = inject(ApiService);
  protected readonly result = signal<BirdResult | null>(null);
  protected readonly showPreview = signal(false);
  protected readonly selectedBird = signal<number | null>(null);

  constructor() {
    effect(onCleanup => {
      const path = this.path();
      const endpoint = '/' + this.labels();
      this.result.set(null);
      this.showPreview.set(false);
      this.selectedBird.set(null);
      const subscription = this.api.get<BirdResult>(endpoint + '/photo', { path }).subscribe({
        next: result => {
          result = { ...result, birds: result.elephants ?? result.birds ?? [] };
          this.result.set(result);
          this.selectedBird.set(result.focus_score_index ?? result.assessment_index ?? result.best_index ?? (result.birds.length ? 0 : null));
        },
        error: () => this.result.set(null),
      });
      onCleanup(() => subscription.unsubscribe());
    });
  }

  protected selectBird(event: Event): void {
    const index = Number((event.target as HTMLSelectElement).value);
    if (Number.isInteger(index) && index >= 0 && index < (this.result()?.birds.length ?? 0)) {
      this.selectedBird.set(index);
    }
  }

  protected preview(index?: number): string {
    const params = new URLSearchParams({ path: this.path() });
    if (index !== undefined) params.set('index', String(index));
    return '/api/' + this.labels() + '/preview?' + params;
  }
}
