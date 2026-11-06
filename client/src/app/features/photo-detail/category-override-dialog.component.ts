import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar } from '@angular/material/snack-bar';
import { firstValueFrom } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';
import { CategoryLabelPipe } from '../gallery/photo-tooltip.component';
import { I18N, I18N_KEYS } from '../../core/i18n/keys';

export type CategoryOverrideDialogData =
  | { path: string; paths?: never; currentCategory: string | null }
  | { paths: string[]; path?: never; currentCategory?: null };

export interface CategoryOverrideResult {
  category: string;
  aggregate: number;
  vcg_submission_score: number | null;
}

export interface BatchCategoryOverrideResult {
  category: string;
  updated: number;
}

interface CategoryOption {
  name: string;
}

/** 编辑模式下设置单张或批量照片的人工分类，后续重新评分仍保留。 */
@Component({
  selector: 'app-category-override-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatDialogModule, MatButtonModule, MatFormFieldModule, MatSelectModule, MatProgressSpinnerModule, TranslatePipe, CategoryLabelPipe],
  template: `
    <h2 mat-dialog-title>{{ I18N.photo.category_override.dialog_title | translate }}</h2>
    <mat-dialog-content class="!pt-2 min-w-[16rem]">
      @if (data.paths) {
        <p class="text-sm mb-3">{{ I18N.photo.category_override.batch_description | translate:{ count: data.paths.length } }}</p>
      }
      @if (loading()) {
        <div class="flex justify-center py-4">
          <mat-spinner diameter="28" [attr.aria-label]="I18N.ui.labels.loading | translate" />
        </div>
      } @else {
        <mat-form-field class="w-full">
          <mat-label>{{ I18N.photo.category_override.label | translate }}</mat-label>
          <mat-select [value]="selectedCategory()" [disabled]="saving()" (selectionChange)="selectedCategory.set($event.value)">
            @for (name of categories(); track name) {
              @let label = ('category_names.' + name) | translate;
              <mat-option [value]="name">{{ label === 'category_names.' + name ? (name | categoryLabel) : label }}</mat-option>
            }
          </mat-select>
        </mat-form-field>
      }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close [disabled]="saving()">{{ I18N.ui.buttons.cancel | translate }}</button>
      <button mat-flat-button [disabled]="loading() || !categories().includes(selectedCategory()) || saving()" (click)="save()">
        {{ saving() ? (I18N.ui.buttons.saving | translate) : (I18N.ui.buttons.save | translate) }}
      </button>
    </mat-dialog-actions>
  `,
})
export class CategoryOverrideDialogComponent implements OnInit {
  protected readonly I18N = I18N_KEYS;
  private readonly api = inject(ApiService);
  private readonly i18n = inject(I18nService);
  private readonly snackBar = inject(MatSnackBar);
  private readonly dialogRef = inject(MatDialogRef<CategoryOverrideDialogComponent>);
  protected readonly data = inject<CategoryOverrideDialogData>(MAT_DIALOG_DATA);

  protected readonly loading = signal(true);
  protected readonly saving = signal(false);
  protected readonly categories = signal<string[]>([]);
  protected readonly selectedCategory = signal(this.data.currentCategory ?? '');

  async ngOnInit(): Promise<void> {
    try {
      const res = await firstValueFrom(this.api.get<{ categories: CategoryOption[] }>('/config/category_priorities'));
      const names = res.categories.map(c => c.name);
      this.categories.set(names);
      if (!this.data.paths && !this.selectedCategory() && names.length) {
        this.selectedCategory.set(names[0]);
      }
    } catch {
      this.snackBar.open(this.i18n.t(I18N.notifications.connection_error), '', { duration: 3000 });
    } finally {
      this.loading.set(false);
    }
  }

  protected async save(): Promise<void> {
    const category = this.selectedCategory();
    if (this.loading() || !this.categories().includes(category) || this.saving()) return;
    this.saving.set(true);
    this.dialogRef.disableClose = true;
    try {
      if (this.data.paths) {
        const res = await firstValueFrom(
          this.api.post<{ new_category: string; updated: number }>('/comparison/override_categories', { paths: this.data.paths, category }),
        );
        this.snackBar.open(this.i18n.t(I18N.photo.category_override.batch_success, { count: res.updated }), '', { duration: 3000 });
        this.dialogRef.close({ category: res.new_category, updated: res.updated } satisfies BatchCategoryOverrideResult);
        return;
      }
      const res = await firstValueFrom(
        this.api.post<{ new_category: string; aggregate: number; vcg_submission_score: number | null }>('/comparison/override_category', { path: this.data.path, category }),
      );
      this.snackBar.open(this.i18n.t(I18N.photo.category_override.success, { category: res.new_category }), '', { duration: 3000 });
      this.dialogRef.close({ category: res.new_category, aggregate: res.aggregate, vcg_submission_score: res.vcg_submission_score } satisfies CategoryOverrideResult);
    } catch {
      this.snackBar.open(this.i18n.t(I18N.errors.action_failed), '', { duration: 3000 });
      this.saving.set(false);
    } finally {
      this.dialogRef.disableClose = false;
    }
  }
}
